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
CSV_PATH = BASE_DIR / "processed" / "buildings_all_with_crops.csv"
OUTPUT_DIR = BASE_DIR / "training_outputs" / "baseline_resnet50"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_WORKERS = 2
USE_CLASS_WEIGHTS = True

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
        x = np.transpose(x, (2, 0, 1))  # (6, H, W)

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
    macro_f1 = f1_score(targets_all, preds_all, average="macro")
    per_class_f1 = f1_score(targets_all, preds_all, average=None, labels=[0, 1, 2, 3])

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
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)

    required_splits = {"train", "test", "hold"}
    found_splits = set(df["split"].unique())
    missing = required_splits - found_splits
    if missing:
        raise ValueError(f"Missing required splits: {missing}")

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    hold_df = df[df["split"] == "hold"].copy()

    print("\nSplit sizes:")
    print(df["split"].value_counts())

    print("\nTrain label distribution:")
    print(train_df["damage_label"].value_counts())

    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": False,
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
        print("\nUsing class weights:", weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state = None
    best_f1 = -1.0
    history = []

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True
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
            model, test_loader, criterion, device, desc=f"Test eval epoch {epoch + 1}"
        )
        epoch_minutes = (time.time() - epoch_start) / 60.0

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "test_loss": test_metrics["loss"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_f1_no_damage": float(test_metrics["per_class_f1"][0]),
                "test_f1_minor": float(test_metrics["per_class_f1"][1]),
                "test_f1_major": float(test_metrics["per_class_f1"][2]),
                "test_f1_destroyed": float(test_metrics["per_class_f1"][3]),
                "epoch_minutes": epoch_minutes,
            }
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} | "
            f"Test F1: {test_metrics['macro_f1']:.4f} | "
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

    final_test = evaluate(model, test_loader, criterion, device, desc="Final TEST evaluation")
    final_hold = evaluate(model, hold_loader, criterion, device, desc="Final HOLD evaluation")

    print(f"\nFinal TEST Macro F1: {final_test['macro_f1']:.4f}")
    print(
        classification_report(
            final_test["targets"],
            final_test["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
        )
    )

    print(f"\nFinal HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            target_names=[IDX_TO_LABEL[i] for i in range(4)],
            digits=4,
        )
    )

    test_report = classification_report(
        final_test["targets"],
        final_test["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
    )
    hold_report = classification_report(
        final_hold["targets"],
        final_hold["preds"],
        target_names=[IDX_TO_LABEL[i] for i in range(4)],
        digits=4,
        output_dict=True,
    )

    # Save predictions for later analysis
    np.save(OUTPUT_DIR / "test_preds.npy", np.array(final_test["preds"]))
    np.save(OUTPUT_DIR / "test_targets.npy", np.array(final_test["targets"]))
    np.save(OUTPUT_DIR / "hold_preds.npy", np.array(final_hold["preds"]))
    np.save(OUTPUT_DIR / "hold_targets.npy", np.array(final_hold["targets"]))

    print("\nSaved TEST and HOLD predictions for confusion matrix analysis.")

    summary = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "device": str(device),
        "best_test_macro_f1": float(best_f1),
        "final_test_macro_f1": float(final_test["macro_f1"]),
        "final_hold_macro_f1": float(final_hold["macro_f1"]),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "hold_size": int(len(hold_df)),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved all outputs successfully.")


if __name__ == "__main__":
    main()