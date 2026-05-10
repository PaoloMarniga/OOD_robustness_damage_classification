from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# =========================
# Configuration
# =========================
BASE_DIR = Path.home() / "Desktop"

CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "baseline_resnet50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
USE_CLASS_WEIGHTS = True

TRAIN_SPLIT = "OOD_train"
TEST_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"

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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =========================
# Device selection
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =========================
# Dataset
# =========================
class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        y = LABEL_TO_IDX[row["damage_label"]]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


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
        progress_bar = tqdm(loader, desc=desc, leave=False)

        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            preds_all.extend(preds.cpu().numpy())
            targets_all.extend(y.cpu().numpy())

    loss_avg = total_loss / len(loader.dataset)

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

    return {
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
    }


# =========================
# Main
# =========================
def main():
    print("Loading OOD data...")
    df = pd.read_csv(CSV_PATH)

    required_splits = {TRAIN_SPLIT, TEST_SPLIT, HOLD_SPLIT}
    found_splits = set(df["split"].unique())

    missing = required_splits - found_splits

    if missing:
        raise ValueError(f"Missing required OOD splits: {missing}")

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == TRAIN_SPLIT].copy()
    test_df = df[df["split"] == TEST_SPLIT].copy()
    hold_df = df[df["split"] == HOLD_SPLIT].copy()

    print("\nSplit sizes:")
    print(df["split"].value_counts())

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    print("\nTest label distribution:")
    print(test_df["damage_label"].value_counts())

    print("\nHold label distribution:")
    print(hold_df["damage_label"].value_counts())

    # Extra OOD leakage check
    train_locations = set(train_df["disaster"])
    test_locations = set(test_df["disaster"])
    hold_locations = set(hold_df["disaster"])

    train_ids = set(train_df["image_id"])
    test_ids = set(test_df["image_id"])
    hold_ids = set(hold_df["image_id"])

    print("\nImage overlap check:")
    print("OOD_train ∩ OOD_test:", len(train_ids & test_ids))
    print("OOD_train ∩ OOD_hold:", len(train_ids & hold_ids))
    print("OOD_test ∩ OOD_hold:", len(test_ids & hold_ids))

    print("\nLocation overlap check:")
    print("OOD_train ∩ OOD_test:", len(train_locations & test_locations))
    print("OOD_train ∩ OOD_hold:", len(train_locations & hold_locations))
    print("OOD_test ∩ OOD_hold:", len(test_locations & hold_locations))

    assert len(train_ids & test_ids) == 0
    assert len(train_ids & hold_ids) == 0
    assert len(test_ids & hold_ids) == 0

    assert len(train_locations & test_locations) == 0
    assert len(train_locations & hold_locations) == 0
    assert len(test_locations & hold_locations) == 0

    print("\nPASS: no image or location overlap.")

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        XViewBuildingDataset(train_df),
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df),
        shuffle=False,
        **loader_kwargs,
    )

    device = get_device()
    print(f"\nUsing device: {device}")

    model = ResNet50SixChannel(num_classes=4).to(device)

    if USE_CLASS_WEIGHTS:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2, 3]),
            y=train_df["damage_label"].map(LABEL_TO_IDX).values,
        )

        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        print("\nUsing class weights:")
        print(weights.cpu().numpy())

    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_f1 = -1.0
    history = []

    print("\nStarting OOD training...")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True,
        )

        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / len(train_loader.dataset)

        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            desc=f"OOD test eval epoch {epoch + 1}",
        )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        history.append(
            {
                "epoch": epoch + 1,
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
            f"Epoch {epoch + 1:02d} | "
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

    print("\nEvaluating best model...")
    model.load_state_dict(best_state)

    final_test = evaluate(
        model,
        test_loader,
        criterion,
        device,
        desc="Final OOD TEST evaluation",
    )

    final_hold = evaluate(
        model,
        hold_loader,
        criterion,
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

    np.save(OUTPUT_DIR / "ood_test_preds.npy", np.array(final_test["preds"]))
    np.save(OUTPUT_DIR / "ood_test_targets.npy", np.array(final_test["targets"]))
    np.save(OUTPUT_DIR / "ood_hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "ood_hold_targets.npy", np.array(final_hold["targets"]))

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "device": str(device),
        "best_ood_test_macro_f1": float(best_f1),
        "final_ood_test_macro_f1": float(final_test["macro_f1"]),
        "final_ood_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
        "train_locations": sorted(list(train_locations)),
        "test_locations": sorted(list(test_locations)),
        "hold_locations": sorted(list(hold_locations)),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "ood_test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "ood_hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved all OOD training outputs successfully.")


if __name__ == "__main__":
    main()