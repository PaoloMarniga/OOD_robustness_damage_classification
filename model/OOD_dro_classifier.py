from pathlib import Path
import random
import json
import copy
import time
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

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "baseline_resnet50_dro"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

TRAIN_SPLIT = "OOD_train"
TEST_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"

UNKNOWN_GROUP = "__unknown__"

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# Device
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =========================
# Dataset with GROUP
# =========================
class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, group_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.group_to_idx = group_to_idx
        self.unknown_group_id = group_to_idx[UNKNOWN_GROUP]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        y = LABEL_TO_IDX[row["damage_label"]]

        group = row["disaster"]
        group_id = self.group_to_idx.get(group, self.unknown_group_id)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(group_id, dtype=torch.long),
        )


# =========================
# Model
# =========================
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
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight

        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# =========================
# Evaluation
# =========================
def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()

    total_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        progress = tqdm(loader, desc=desc, leave=False)

        for x, y, _ in progress:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    macro_f1 = f1_score(
        targets_all,
        preds_all,
        average="macro",
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets_all,
        preds_all,
        average=None,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )

    pred_counts = pd.Series(preds_all).value_counts().sort_index().to_dict()
    target_counts = pd.Series(targets_all).value_counts().sort_index().to_dict()

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
        "pred_counts": pred_counts,
        "target_counts": target_counts,
    }


def print_prediction_distribution(metrics, title):
    print(f"\n{title} prediction distribution:")
    for idx in range(4):
        print(f"{IDX_TO_LABEL[idx]}: {metrics['pred_counts'].get(idx, 0)}")

    print(f"\n{title} true distribution:")
    for idx in range(4):
        print(f"{IDX_TO_LABEL[idx]}: {metrics['target_counts'].get(idx, 0)}")


# =========================
# MAIN
# =========================
def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == TRAIN_SPLIT].copy()
    test_df = df[df["split"] == TEST_SPLIT].copy()
    hold_df = df[df["split"] == HOLD_SPLIT].copy()

    if train_df.empty:
        raise ValueError(f"No rows found for split: {TRAIN_SPLIT}")
    if test_df.empty:
        raise ValueError(f"No rows found for split: {TEST_SPLIT}")
    if hold_df.empty:
        raise ValueError(f"No rows found for split: {HOLD_SPLIT}")

    print("\nSplit sizes:")
    print("Train:", len(train_df))
    print("Test:", len(test_df))
    print("Hold:", len(hold_df))

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["damage_label"].value_counts())

    print("\nHold label distribution:")
    print(hold_df["damage_label"].value_counts())

    # =========================
    # Group mapping
    # =========================
    unique_train_groups = sorted(train_df["disaster"].unique())

    group_to_idx = {g: i for i, g in enumerate(unique_train_groups)}
    group_to_idx[UNKNOWN_GROUP] = len(group_to_idx)

    print("\nNumber of training groups:", len(unique_train_groups))
    print("Group mapping:")
    for group_name, group_id in group_to_idx.items():
        print(f"  {group_id}: {group_name}")

    unseen_test_groups = sorted(set(test_df["disaster"].unique()) - set(unique_train_groups))
    unseen_hold_groups = sorted(set(hold_df["disaster"].unique()) - set(unique_train_groups))

    print("\nUnseen TEST groups mapped to __unknown__:", unseen_test_groups)
    print("Unseen HOLD groups mapped to __unknown__:", unseen_hold_groups)

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewBuildingDataset(train_df, group_to_idx),
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df, group_to_idx),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df, group_to_idx),
        shuffle=False,
        **loader_kwargs,
    )

    device = get_device()
    print("\nUsing device:", device)

    model = ResNet50SixChannel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # This criterion is only used for evaluation loss.
    # Training uses the DRO-lite group-weighted loss.
    eval_criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    history = []

    print("\nStarting DRO-lite training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_start = time.time()

        total_loss = 0.0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS}",
            leave=True,
        )

        for x, y, g in progress:
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            optimizer.zero_grad()

            logits = model(x)

            # Per-sample cross entropy loss
            losses = F.cross_entropy(logits, y, reduction="none")

            # DRO-lite over groups present in the batch
            group_losses = []
            unique_g = g.unique()

            for group_id in unique_g:
                mask = g == group_id
                group_loss = losses[mask].mean()
                group_losses.append(group_loss)

            group_losses = torch.stack(group_losses)

            # Higher-loss groups receive higher weight
            weights = group_losses / (group_losses.sum() + 1e-8)

            loss = (weights * group_losses).sum()

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during DRO-lite training.")

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_samples

        test_metrics = evaluate(
            model,
            test_loader,
            eval_criterion,
            device,
            desc=f"OOD test eval epoch {epoch}",
        )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "ood_test_loss": test_metrics["loss"],
                "ood_test_macro_f1": test_metrics["macro_f1"],
                "ood_test_f1_no_damage": float(test_metrics["per_class_f1"][0]),
                "ood_test_f1_minor": float(test_metrics["per_class_f1"][1]),
                "ood_test_f1_major": float(test_metrics["per_class_f1"][2]),
                "ood_test_f1_destroyed": float(test_metrics["per_class_f1"][3]),
                "epoch_minutes": epoch_minutes,
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"OOD Test Loss: {test_metrics['loss']:.4f} | "
            f"OOD Test Macro F1: {test_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        if test_metrics["macro_f1"] > best_f1:
            best_f1 = test_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best model state was saved.")

    torch.save(best_state, OUTPUT_DIR / "best_model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # =========================
    # Final evaluation
    # =========================
    print("\nEvaluating best DRO-lite model...")
    model.load_state_dict(best_state)

    final_test = evaluate(
        model,
        test_loader,
        eval_criterion,
        device,
        desc="Final OOD TEST evaluation",
    )

    final_hold = evaluate(
        model,
        hold_loader,
        eval_criterion,
        device,
        desc="Final OOD HOLD evaluation",
    )

    print(f"\nFinal OOD TEST Macro F1: {final_test['macro_f1']:.4f}")
    print(
        classification_report(
            final_test["targets"],
            final_test["preds"],
            labels=[0, 1, 2, 3],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
            zero_division=0,
        )
    )
    print_prediction_distribution(final_test, "OOD TEST")

    print(f"\nFinal OOD HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            labels=[0, 1, 2, 3],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
            zero_division=0,
        )
    )
    print_prediction_distribution(final_hold, "OOD HOLD")

    test_report = classification_report(
        final_test["targets"],
        final_test["preds"],
        labels=[0, 1, 2, 3],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    hold_report = classification_report(
        final_hold["targets"],
        final_hold["preds"],
        labels=[0, 1, 2, 3],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    test_cm = confusion_matrix(
        final_test["targets"],
        final_test["preds"],
        labels=[0, 1, 2, 3],
    )

    hold_cm = confusion_matrix(
        final_hold["targets"],
        final_hold["preds"],
        labels=[0, 1, 2, 3],
    )

    np.save(OUTPUT_DIR / "ood_test_preds.npy", np.array(final_test["preds"]))
    np.save(OUTPUT_DIR / "ood_test_targets.npy", np.array(final_test["targets"]))
    np.save(OUTPUT_DIR / "ood_hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "ood_hold_targets.npy", np.array(final_hold["targets"]))
    np.save(OUTPUT_DIR / "ood_test_confusion_matrix.npy", test_cm)
    np.save(OUTPUT_DIR / "ood_hold_confusion_matrix.npy", hold_cm)

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "method": "DRO-lite group-weighted objective over disaster locations",
        "group_definition": "disaster location",
        "training_groups": unique_train_groups,
        "unknown_group": UNKNOWN_GROUP,
        "unseen_test_groups": unseen_test_groups,
        "unseen_hold_groups": unseen_hold_groups,
        "best_ood_test_macro_f1": float(best_f1),
        "final_ood_test_macro_f1": float(final_test["macro_f1"]),
        "final_ood_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
        "test_prediction_distribution": {
            IDX_TO_LABEL[int(k)]: int(v) for k, v in final_test["pred_counts"].items()
        },
        "hold_prediction_distribution": {
            IDX_TO_LABEL[int(k)]: int(v) for k, v in final_hold["pred_counts"].items()
        },
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "ood_test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "ood_hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved all DRO-lite outputs successfully.")
    print(OUTPUT_DIR / "best_model.pt")
    print(OUTPUT_DIR / "training_history.csv")
    print(OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()