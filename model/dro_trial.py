from pathlib import Path
import random, json, copy, time
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# =========================
# Configuration
# =========================

BASE_DIR = Path.home() / "Desktop"
CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "resnet50_dro_1seed_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

USE_CLASS_WEIGHTS = False

DRO_ETA = 1.0
DRO_AVG_RISK_WEIGHT = 0.5
DRO_WORST_RISK_WEIGHT = 0.5

TRAIN_SPLIT = "OOD_train"
VAL_SPLIT = "OOD_test"
FINAL_TEST_SPLIT = "OOD_hold"

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
LABEL_IDS = [0, 1, 2, 3]

IMAGENET_MEAN_6 = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_6 = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, group_to_idx, env_col):
        self.df = dataframe.reset_index(drop=True)
        self.group_to_idx = group_to_idx
        self.env_col = env_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])

        if x.ndim != 3 or x.shape[2] != 6:
            raise ValueError(f"Expected crop shape H,W,6. Got {x.shape}")

        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = (x - IMAGENET_MEAN_6[:, None, None]) / IMAGENET_STD_6[:, None, None]

        y = LABEL_TO_IDX[row["damage_label"]]
        g = self.group_to_idx[row[self.env_col]]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(g, dtype=torch.long),
        )


class ResNet50SixChannel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.backbone = resnet50(weights=weights)

        old_conv = self.backbone.conv1

        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight * 0.5
            new_conv.weight[:, 3:, :, :] = old_conv.weight * 0.5

        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def dro_group_loss(logits, y, g):
    per_sample_losses = F.cross_entropy(logits, y, reduction="none")

    group_losses = []

    for group_id in g.unique():
        mask = g == group_id
        group_losses.append(per_sample_losses[mask].mean())

    group_losses = torch.stack(group_losses)

    avg_risk = group_losses.mean()
    group_weights = torch.softmax(DRO_ETA * group_losses.detach(), dim=0)
    worst_weighted_risk = (group_weights * group_losses).sum()

    loss = DRO_AVG_RISK_WEIGHT * avg_risk + DRO_WORST_RISK_WEIGHT * worst_weighted_risk

    return loss


def evaluate(model, loader, criterion, device, desc):
    model.eval()

    total_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc=desc, leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(targets_all, preds_all, average="macro", labels=LABEL_IDS, zero_division=0)
    per_class_f1 = f1_score(targets_all, preds_all, average=None, labels=LABEL_IDS, zero_division=0)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
    }


def main():
    set_seed(SEED)

    print("Loading OOD data...")
    df = pd.read_csv(CSV_PATH)
    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    env_col = "disaster"
    if env_col not in df.columns:
        raise ValueError("Expected a disaster column for DRO groups.")

    train_df = df[df["split"] == TRAIN_SPLIT].copy().reset_index(drop=True)
    val_df = df[df["split"] == VAL_SPLIT].copy().reset_index(drop=True)
    hold_df = df[df["split"] == FINAL_TEST_SPLIT].copy().reset_index(drop=True)

    print("\nSplit sizes:")
    print(df["split"].value_counts())

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    print("\nOOD validation label distribution:")
    print(val_df["damage_label"].value_counts())

    print("\nOOD hold label distribution:")
    print(hold_df["damage_label"].value_counts())

    train_envs = sorted(train_df[env_col].unique())
    val_envs = sorted(val_df[env_col].unique())
    hold_envs = sorted(hold_df[env_col].unique())

    print("\nTrain environments:", train_envs)
    print("OOD validation environments:", val_envs)
    print("OOD hold environments:", hold_envs)

    assert len(set(train_df[env_col]) & set(val_df[env_col])) == 0
    assert len(set(train_df[env_col]) & set(hold_df[env_col])) == 0
    assert len(set(val_df[env_col]) & set(hold_df[env_col])) == 0

    print("\nPASS: no location overlap.")

    group_to_idx = {g: i for i, g in enumerate(train_envs)}

    device = get_device()
    print(f"\nUsing device: {device}")

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewBuildingDataset(train_df, group_to_idx, env_col),
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        XViewBuildingDataset(val_df, {g: 0 for g in val_envs}, env_col),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df, {g: 0 for g in hold_envs}, env_col),
        shuffle=False,
        **loader_kwargs,
    )

    model = ResNet50SixChannel(num_classes=4).to(device)

    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_val_f1 = -1.0
    best_epoch = None
    history = []

    print("\nStarting one seed DRO training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        total_loss = 0.0
        total_samples = 0

        for x, y, g in tqdm(train_loader, desc=f"DRO Epoch {epoch}/{NUM_EPOCHS}", leave=True):
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = dro_group_loss(logits, y, g)

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss at epoch {epoch}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        train_loss = total_loss / total_samples

        val_metrics = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
            desc=f"OOD validation eval epoch {epoch}",
        )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "ood_val_loss": val_metrics["loss"],
            "ood_val_macro_f1": val_metrics["macro_f1"],
            "ood_val_f1_no_damage": float(val_metrics["per_class_f1"][0]),
            "ood_val_f1_minor": float(val_metrics["per_class_f1"][1]),
            "ood_val_f1_major": float(val_metrics["per_class_f1"][2]),
            "ood_val_f1_destroyed": float(val_metrics["per_class_f1"][3]),
            "epoch_minutes": epoch_minutes,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"OOD Val Loss: {val_metrics['loss']:.4f} | "
            f"OOD Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    if best_state is None:
        raise RuntimeError("No best model was saved.")

    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

    print(f"\nEvaluating best epoch: {best_epoch}")
    model.load_state_dict(best_state)

    final_val = evaluate(model, val_loader, eval_criterion, device, "Final OOD validation evaluation")
    final_hold = evaluate(model, hold_loader, eval_criterion, device, "Final OOD hold evaluation")

    print(f"\nFinal OOD VAL Macro F1: {final_val['macro_f1']:.4f}")
    print(classification_report(
        final_val["targets"],
        final_val["preds"],
        labels=LABEL_IDS,
        target_names=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        digits=4,
        zero_division=0,
    ))

    print(f"\nFinal OOD HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(classification_report(
        final_hold["targets"],
        final_hold["preds"],
        labels=LABEL_IDS,
        target_names=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        digits=4,
        zero_division=0,
    ))

    np.save(OUTPUT_DIR / "ood_val_preds.npy", np.array(final_val["preds"]))
    np.save(OUTPUT_DIR / "ood_val_targets.npy", np.array(final_val["targets"]))
    np.save(OUTPUT_DIR / "ood_hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "ood_hold_targets.npy", np.array(final_hold["targets"]))

    np.save(
        OUTPUT_DIR / "ood_val_confusion_matrix.npy",
        confusion_matrix(final_val["targets"], final_val["preds"], labels=LABEL_IDS),
    )

    np.save(
        OUTPUT_DIR / "ood_hold_confusion_matrix.npy",
        confusion_matrix(final_hold["targets"], final_hold["preds"], labels=LABEL_IDS),
    )

    summary = {
        "method": "resnet50_dro_1seed_test",
        "seed": SEED,
        "best_epoch": best_epoch,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "dro_eta": DRO_ETA,
        "dro_avg_risk_weight": DRO_AVG_RISK_WEIGHT,
        "dro_worst_risk_weight": DRO_WORST_RISK_WEIGHT,
        "ood_val_macro_f1": float(final_val["macro_f1"]),
        "ood_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "ood_val_size": int(len(val_df)),
        "ood_hold_size": int(len(hold_df)),
        "train_environments": train_envs,
        "ood_val_environments": val_envs,
        "ood_hold_environments": hold_envs,
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved one seed DRO outputs successfully.")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()