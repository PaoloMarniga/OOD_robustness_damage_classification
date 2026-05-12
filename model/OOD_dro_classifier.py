from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
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

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "baseline_resnet50_dro_5seeds_1se"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 999, 2024, 2025]

BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

USE_CLASS_WEIGHTS = True

DRO_ETA = 1.0
DRO_AVG_RISK_WEIGHT = 0.5
DRO_WORST_RISK_WEIGHT = 0.5

TRAIN_SPLIT = "OOD_train"
VAL_SPLIT = "OOD_test"
FINAL_TEST_SPLIT = "OOD_hold"

UNKNOWN_GROUP = "__unknown__"

ENV_COLUMN_CANDIDATES = [
    "disaster",
    "location",
    "event",
    "environment",
]

LABEL_TO_IDX = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}
LABEL_IDS = [0, 1, 2, 3]

IMAGENET_MEAN_6 = np.array(
    [0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
    dtype=np.float32,
)

IMAGENET_STD_6 = np.array(
    [0.229, 0.224, 0.225, 0.229, 0.224, 0.225],
    dtype=np.float32,
)


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
# Utility functions
# =========================

def find_environment_column(df: pd.DataFrame):
    for col in ENV_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def compute_train_class_weights(train_df):
    y_train = train_df["damage_label"].map(LABEL_TO_IDX).values
    present_classes = set(y_train.tolist())
    missing_classes = set(LABEL_IDS) - present_classes

    if missing_classes:
        raise ValueError(f"Missing classes in training data: {missing_classes}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(LABEL_IDS),
        y=y_train,
    )

    return weights.astype(np.float32)


# =========================
# Dataset
# =========================

class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, group_to_idx, env_col):
        self.df = dataframe.reset_index(drop=True)
        self.group_to_idx = group_to_idx
        self.env_col = env_col
        self.unknown_group_id = group_to_idx[UNKNOWN_GROUP]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])

        if x.ndim != 3:
            raise ValueError(f"Expected crop with shape H,W,C. Got shape {x.shape}")

        if x.shape[2] != 6:
            raise ValueError(f"Expected 6 channels. Got shape {x.shape}")

        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        x = (x - IMAGENET_MEAN_6[:, None, None]) / IMAGENET_STD_6[:, None, None]

        y = LABEL_TO_IDX[row["damage_label"]]

        group_name = row[self.env_col]
        group_id = self.group_to_idx.get(group_name, self.unknown_group_id)

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
            new_conv.weight[:, :3, :, :] = old_conv.weight * 0.5
            new_conv.weight[:, 3:, :, :] = old_conv.weight * 0.5

        self.backbone.conv1 = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# =========================
# DRO objective
# =========================

def dro_group_loss(logits, y, g, class_weights=None):
    per_sample_losses = F.cross_entropy(
        logits,
        y,
        weight=class_weights,
        reduction="none",
    )

    group_losses = []
    group_ids = []

    for group_id in g.unique():
        mask = g == group_id
        group_losses.append(per_sample_losses[mask].mean())
        group_ids.append(int(group_id.item()))

    group_losses = torch.stack(group_losses)

    avg_risk = group_losses.mean()

    group_weights = torch.softmax(DRO_ETA * group_losses.detach(), dim=0)
    worst_weighted_risk = (group_weights * group_losses).sum()

    loss = (
        DRO_AVG_RISK_WEIGHT * avg_risk
        + DRO_WORST_RISK_WEIGHT * worst_weighted_risk
    )

    return loss, group_losses.detach(), group_weights.detach(), group_ids


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
        labels=LABEL_IDS,
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets_all,
        preds_all,
        average=None,
        labels=LABEL_IDS,
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


def classification_report_dict(targets, preds):
    return classification_report(
        targets,
        preds,
        labels=LABEL_IDS,
        target_names=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        digits=4,
        output_dict=True,
        zero_division=0,
    )


def classification_report_text(targets, preds):
    return classification_report(
        targets,
        preds,
        labels=LABEL_IDS,
        target_names=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        digits=4,
        zero_division=0,
    )


def compute_per_class_table(metrics, split_name, seed, method_name):
    rows = []

    for idx in LABEL_IDS:
        rows.append(
            {
                "method": method_name,
                "seed": seed,
                "split": split_name,
                "class_id": idx,
                "class_name": IDX_TO_LABEL[idx],
                "f1": float(metrics["per_class_f1"][idx]),
                "true_count": int(metrics["target_counts"].get(idx, 0)),
                "pred_count": int(metrics["pred_counts"].get(idx, 0)),
            }
        )

    return pd.DataFrame(rows)


def compute_per_environment_table(
    dataframe,
    preds,
    targets,
    env_col,
    split_name,
    seed,
    method_name,
):
    temp = dataframe.reset_index(drop=True).copy()
    temp["target"] = targets
    temp["pred"] = preds

    rows = []

    for env, group in temp.groupby(env_col):
        macro_f1 = f1_score(
            group["target"],
            group["pred"],
            average="macro",
            labels=LABEL_IDS,
            zero_division=0,
        )

        per_class = f1_score(
            group["target"],
            group["pred"],
            average=None,
            labels=LABEL_IDS,
            zero_division=0,
        )

        row = {
            "method": method_name,
            "seed": seed,
            "split": split_name,
            "environment": env,
            "n": int(len(group)),
            "macro_f1": float(macro_f1),
        }

        for idx in LABEL_IDS:
            name = IDX_TO_LABEL[idx].replace("-", "_")
            row[f"f1_{name}"] = float(per_class[idx])
            row[f"true_{name}"] = int((group["target"] == idx).sum())
            row[f"pred_{name}"] = int((group["pred"] == idx).sum())

        rows.append(row)

    return pd.DataFrame(rows).sort_values("macro_f1")


def save_prediction_dataframe(dataframe, preds, targets, output_path, env_col):
    pred_df = dataframe.reset_index(drop=True).copy()
    pred_df["target_id"] = targets
    pred_df["pred_id"] = preds
    pred_df["target_label"] = [IDX_TO_LABEL[int(x)] for x in targets]
    pred_df["pred_label"] = [IDX_TO_LABEL[int(x)] for x in preds]
    pred_df["correct"] = pred_df["target_id"] == pred_df["pred_id"]

    keep_cols = []

    for col in [
        "split",
        env_col,
        "image_id",
        "building_id",
        "crop_path",
        "damage_label",
        "target_id",
        "target_label",
        "pred_id",
        "pred_label",
        "correct",
    ]:
        if col is not None and col in pred_df.columns and col not in keep_cols:
            keep_cols.append(col)

    pred_df[keep_cols].to_csv(output_path, index=False)


# =========================
# OOD leakage checks
# =========================

def run_ood_leakage_checks(train_df, val_df, hold_df, env_col):
    if "image_id" in train_df.columns:
        train_ids = set(train_df["image_id"])
        val_ids = set(val_df["image_id"])
        hold_ids = set(hold_df["image_id"])

        print("\nImage overlap check:")
        print("OOD_train intersection OOD_val:", len(train_ids & val_ids))
        print("OOD_train intersection OOD_hold:", len(train_ids & hold_ids))
        print("OOD_val intersection OOD_hold:", len(val_ids & hold_ids))

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & hold_ids) == 0
        assert len(val_ids & hold_ids) == 0

    train_locations = set(train_df[env_col])
    val_locations = set(val_df[env_col])
    hold_locations = set(hold_df[env_col])

    print("\nLocation overlap check:")
    print("OOD_train intersection OOD_val:", len(train_locations & val_locations))
    print("OOD_train intersection OOD_hold:", len(train_locations & hold_locations))
    print("OOD_val intersection OOD_hold:", len(val_locations & hold_locations))

    assert len(train_locations & val_locations) == 0
    assert len(train_locations & hold_locations) == 0
    assert len(val_locations & hold_locations) == 0

    print("\nPASS: no image or location overlap.")


# =========================
# Training one seed
# =========================

def train_one_seed(
    seed,
    train_df,
    train_loader,
    val_loader,
    device,
    group_to_idx,
):
    set_seed(seed)

    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    checkpoint_dir = seed_dir / "checkpoints"

    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Starting DRO OOD seed {seed}")
    print("=" * 80)

    model = ResNet50SixChannel(num_classes=4).to(device)

    if USE_CLASS_WEIGHTS:
        class_weights_np = compute_train_class_weights(train_df)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        print("\nUsing class weights:")
        print(class_weights.cpu().numpy())
    else:
        class_weights_np = None
        class_weights = None
        print("\nUsing unweighted cross entropy.")

    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []

    print("\nStarting DRO training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Seed {seed} | DRO Epoch {epoch}/{NUM_EPOCHS}",
            leave=True,
        )

        for x, y, g in progress:
            x = x.to(device)
            y = y.to(device)
            g = g.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss, group_losses, group_weights, group_ids = dro_group_loss(
                logits,
                y,
                g,
                class_weights=class_weights,
            )

            if torch.isnan(loss):
                raise RuntimeError(f"NaN loss detected at seed {seed}, epoch {epoch}.")

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_loss / total_samples

        val_metrics = evaluate(
            model,
            val_loader,
            eval_criterion,
            device,
            desc=f"Seed {seed} | OOD validation eval epoch {epoch}",
        )

        epoch_minutes = (time.time() - epoch_start) / 60.0

        history.append(
            {
                "seed": seed,
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
        )

        print(
            f"Seed {seed} | Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"OOD Val Loss: {val_metrics['loss']:.4f} | "
            f"OOD Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        torch.save(
            {
                "seed": seed,
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": optimizer.state_dict(),
                "ood_val_macro_f1": float(val_metrics["macro_f1"]),
                "ood_val_loss": float(val_metrics["loss"]),
                "class_weights": class_weights_np,
                "group_to_idx": group_to_idx,
            },
            checkpoint_dir / f"epoch_{epoch:02d}.pt",
        )

    history_df = pd.DataFrame(history)
    history_df.to_csv(seed_dir / "training_history.csv", index=False)

    return history_df


# =========================
# 1SE model selection
# =========================

def select_epoch_1se(all_history_df):
    epoch_summary = (
        all_history_df.groupby("epoch")["ood_val_macro_f1"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "ood_val_macro_f1_mean",
                "std": "ood_val_macro_f1_std",
                "count": "num_seeds",
            }
        )
    )

    epoch_summary["ood_val_macro_f1_std"] = epoch_summary["ood_val_macro_f1_std"].fillna(0.0)

    epoch_summary["ood_val_macro_f1_se"] = (
        epoch_summary["ood_val_macro_f1_std"] / np.sqrt(epoch_summary["num_seeds"])
    )

    best_idx = epoch_summary["ood_val_macro_f1_mean"].idxmax()
    best_row = epoch_summary.loc[best_idx]

    best_epoch = int(best_row["epoch"])
    best_mean = float(best_row["ood_val_macro_f1_mean"])
    best_std = float(best_row["ood_val_macro_f1_std"])
    best_se = float(best_row["ood_val_macro_f1_se"])

    threshold = best_mean - best_se

    eligible = epoch_summary[epoch_summary["ood_val_macro_f1_mean"] >= threshold].copy()
    selected_epoch = int(eligible["epoch"].min())

    epoch_summary["one_se_threshold"] = threshold
    epoch_summary["is_best_mean_epoch"] = epoch_summary["epoch"] == best_epoch
    epoch_summary["is_eligible_1se"] = epoch_summary["ood_val_macro_f1_mean"] >= threshold
    epoch_summary["is_selected_1se_epoch"] = epoch_summary["epoch"] == selected_epoch

    selection_info = {
        "best_epoch_by_mean_ood_validation_f1": best_epoch,
        "best_mean_ood_validation_macro_f1": best_mean,
        "std_at_best_epoch": best_std,
        "se_at_best_epoch": best_se,
        "one_se_threshold": threshold,
        "selected_epoch_1se_rule": selected_epoch,
    }

    return selected_epoch, epoch_summary, selection_info


# =========================
# Final evaluation
# =========================

def evaluate_selected_epoch_for_seed(
    seed,
    selected_epoch,
    val_df,
    hold_df,
    val_loader,
    hold_loader,
    device,
    env_col,
):
    seed_dir = OUTPUT_DIR / f"seed_{seed}"
    checkpoint_path = seed_dir / "checkpoints" / f"epoch_{selected_epoch:02d}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    model = ResNet50SixChannel(num_classes=4).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if USE_CLASS_WEIGHTS:
        class_weights_np = checkpoint["class_weights"]
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("\n" + "-" * 80)
    print(f"Evaluating DRO OOD seed {seed}, selected epoch {selected_epoch}")
    print("-" * 80)

    final_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
        desc=f"Seed {seed} | Final OOD validation evaluation",
    )

    final_hold = evaluate(
        model,
        hold_loader,
        criterion,
        device,
        desc=f"Seed {seed} | Final OOD hold evaluation",
    )

    print(f"\nSeed {seed} | Final OOD VAL Macro F1: {final_val['macro_f1']:.4f}")
    print(classification_report_text(final_val["targets"], final_val["preds"]))

    print(f"\nSeed {seed} | Final OOD HOLD Macro F1: {final_hold['macro_f1']:.4f}")
    print(classification_report_text(final_hold["targets"], final_hold["preds"]))

    np.save(seed_dir / "ood_val_preds_selected_1se.npy", np.array(final_val["preds"]))
    np.save(seed_dir / "ood_val_targets_selected_1se.npy", np.array(final_val["targets"]))
    np.save(seed_dir / "ood_hold_preds_selected_1se.npy", np.array(final_hold["preds"]))
    np.save(seed_dir / "ood_hold_targets_selected_1se.npy", np.array(final_hold["targets"]))

    val_cm = confusion_matrix(final_val["targets"], final_val["preds"], labels=LABEL_IDS)
    hold_cm = confusion_matrix(final_hold["targets"], final_hold["preds"], labels=LABEL_IDS)

    np.save(seed_dir / "ood_val_confusion_matrix_selected_1se.npy", val_cm)
    np.save(seed_dir / "ood_hold_confusion_matrix_selected_1se.npy", hold_cm)

    pd.DataFrame(
        val_cm,
        index=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        columns=[IDX_TO_LABEL[i] for i in LABEL_IDS],
    ).to_csv(seed_dir / "ood_val_confusion_matrix_selected_1se.csv")

    pd.DataFrame(
        hold_cm,
        index=[IDX_TO_LABEL[i] for i in LABEL_IDS],
        columns=[IDX_TO_LABEL[i] for i in LABEL_IDS],
    ).to_csv(seed_dir / "ood_hold_confusion_matrix_selected_1se.csv")

    with open(seed_dir / "ood_val_classification_report_selected_1se.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(classification_report_dict(final_val["targets"], final_val["preds"])), f, indent=2)

    with open(seed_dir / "ood_hold_classification_report_selected_1se.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(classification_report_dict(final_hold["targets"], final_hold["preds"])), f, indent=2)

    save_prediction_dataframe(
        val_df,
        final_val["preds"],
        final_val["targets"],
        seed_dir / "ood_val_predictions_selected_1se.csv",
        env_col,
    )

    save_prediction_dataframe(
        hold_df,
        final_hold["preds"],
        final_hold["targets"],
        seed_dir / "ood_hold_predictions_selected_1se.csv",
        env_col,
    )

    val_per_class = compute_per_class_table(
        final_val,
        split_name="ood_validation",
        seed=seed,
        method_name="resnet50_dro",
    )

    hold_per_class = compute_per_class_table(
        final_hold,
        split_name="ood_hold",
        seed=seed,
        method_name="resnet50_dro",
    )

    per_class = pd.concat([val_per_class, hold_per_class], ignore_index=True)
    per_class.to_csv(seed_dir / "per_class_metrics_selected_1se.csv", index=False)

    val_per_env = compute_per_environment_table(
        val_df,
        final_val["preds"],
        final_val["targets"],
        env_col,
        split_name="ood_validation",
        seed=seed,
        method_name="resnet50_dro",
    )

    hold_per_env = compute_per_environment_table(
        hold_df,
        final_hold["preds"],
        final_hold["targets"],
        env_col,
        split_name="ood_hold",
        seed=seed,
        method_name="resnet50_dro",
    )

    if len(val_per_env) > 0:
        val_per_env.to_csv(seed_dir / "ood_val_per_environment_metrics_selected_1se.csv", index=False)

    if len(hold_per_env) > 0:
        hold_per_env.to_csv(seed_dir / "ood_hold_per_environment_metrics_selected_1se.csv", index=False)

    worst_val_env = None
    worst_hold_env = None

    if len(val_per_env) > 0:
        worst_val_env = val_per_env.sort_values("macro_f1").iloc[0].to_dict()

    if len(hold_per_env) > 0:
        worst_hold_env = hold_per_env.sort_values("macro_f1").iloc[0].to_dict()

    result = {
        "seed": seed,
        "selected_epoch_1se": selected_epoch,
        "ood_val_macro_f1": float(final_val["macro_f1"]),
        "ood_hold_macro_f1": float(final_hold["macro_f1"]),
        "ood_val_loss": float(final_val["loss"]),
        "ood_hold_loss": float(final_hold["loss"]),
        "ood_val_f1_no_damage": float(final_val["per_class_f1"][0]),
        "ood_val_f1_minor": float(final_val["per_class_f1"][1]),
        "ood_val_f1_major": float(final_val["per_class_f1"][2]),
        "ood_val_f1_destroyed": float(final_val["per_class_f1"][3]),
        "ood_hold_f1_no_damage": float(final_hold["per_class_f1"][0]),
        "ood_hold_f1_minor": float(final_hold["per_class_f1"][1]),
        "ood_hold_f1_major": float(final_hold["per_class_f1"][2]),
        "ood_hold_f1_destroyed": float(final_hold["per_class_f1"][3]),
        "worst_ood_validation_environment": worst_val_env,
        "worst_ood_hold_environment": worst_hold_env,
    }

    with open(seed_dir / "selected_1se_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(result), f, indent=2)

    return result


# =========================
# Main
# =========================

def main():
    print("Loading OOD data...")
    df = pd.read_csv(CSV_PATH)

    required_splits = {TRAIN_SPLIT, VAL_SPLIT, FINAL_TEST_SPLIT}
    found_splits = set(df["split"].unique())
    missing = required_splits - found_splits

    if missing:
        raise ValueError(f"Missing required OOD splits: {missing}")

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    env_col = find_environment_column(df)

    if env_col is None:
        raise ValueError("No environment column found. DRO requires environment labels.")

    print(f"\nUsing environment column for DRO groups: {env_col}")

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

    print("\nTrain environments:")
    print(sorted(train_df[env_col].unique()))

    print("\nOOD validation environments:")
    print(sorted(val_df[env_col].unique()))

    print("\nOOD hold environments:")
    print(sorted(hold_df[env_col].unique()))

    run_ood_leakage_checks(train_df, val_df, hold_df, env_col)

    unique_train_groups = sorted(train_df[env_col].unique())

    group_to_idx = {g: i for i, g in enumerate(unique_train_groups)}
    group_to_idx[UNKNOWN_GROUP] = len(group_to_idx)

    unseen_val_groups = sorted(set(val_df[env_col].unique()) - set(unique_train_groups))
    unseen_hold_groups = sorted(set(hold_df[env_col].unique()) - set(unique_train_groups))

    print("\nNumber of training groups:", len(unique_train_groups))
    print("Group mapping:")
    for group_name, group_id in group_to_idx.items():
        print(f"  {group_id}: {group_name}")

    print("\nUnseen OOD validation groups mapped to __unknown__:", unseen_val_groups)
    print("Unseen OOD hold groups mapped to __unknown__:", unseen_hold_groups)

    device = get_device()
    print("\nUsing device:", device)

    all_histories = []

    for seed in SEEDS:
        set_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

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
            XViewBuildingDataset(val_df, group_to_idx, env_col),
            shuffle=False,
            **loader_kwargs,
        )

        history_df = train_one_seed(
            seed=seed,
            train_df=train_df,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            group_to_idx=group_to_idx,
        )

        all_histories.append(history_df)

    all_history_df = pd.concat(all_histories, ignore_index=True)
    all_history_df.to_csv(OUTPUT_DIR / "all_seed_training_history.csv", index=False)

    selected_epoch, epoch_summary, selection_info = select_epoch_1se(all_history_df)

    epoch_summary.to_csv(OUTPUT_DIR / "epoch_ood_validation_summary_1se_rule.csv", index=False)

    with open(OUTPUT_DIR / "model_selection_1se_rule.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(selection_info), f, indent=2)

    print("\n" + "=" * 80)
    print("1SE model selection")
    print("=" * 80)
    print(json.dumps(selection_info, indent=2))

    final_results = []

    for seed in SEEDS:
        set_seed(seed)

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader_kwargs = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": seed_worker,
            "generator": generator,
        }

        if NUM_WORKERS > 0:
            loader_kwargs["persistent_workers"] = True

        val_loader = DataLoader(
            XViewBuildingDataset(val_df, group_to_idx, env_col),
            shuffle=False,
            **loader_kwargs,
        )

        hold_loader = DataLoader(
            XViewBuildingDataset(hold_df, group_to_idx, env_col),
            shuffle=False,
            **loader_kwargs,
        )

        result = evaluate_selected_epoch_for_seed(
            seed=seed,
            selected_epoch=selected_epoch,
            val_df=val_df,
            hold_df=hold_df,
            val_loader=val_loader,
            hold_loader=hold_loader,
            device=device,
            env_col=env_col,
        )

        final_results.append(result)

    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv(OUTPUT_DIR / "final_results_by_seed_selected_1se.csv", index=False)

    metric_cols = [
        "ood_val_macro_f1",
        "ood_hold_macro_f1",
        "ood_val_loss",
        "ood_hold_loss",
        "ood_val_f1_no_damage",
        "ood_val_f1_minor",
        "ood_val_f1_major",
        "ood_val_f1_destroyed",
        "ood_hold_f1_no_damage",
        "ood_hold_f1_minor",
        "ood_hold_f1_major",
        "ood_hold_f1_destroyed",
    ]

    aggregate_rows = []

    for metric in metric_cols:
        values = final_results_df[metric].astype(float)

        aggregate_rows.append(
            {
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "se": float(values.std(ddof=1) / np.sqrt(values.count())),
                "min": float(values.min()),
                "max": float(values.max()),
                "num_seeds": int(values.count()),
            }
        )

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(OUTPUT_DIR / "final_results_mean_std_se_selected_1se.csv", index=False)

    per_class_all = []

    for seed in SEEDS:
        path = OUTPUT_DIR / f"seed_{seed}" / "per_class_metrics_selected_1se.csv"
        if path.exists():
            per_class_all.append(pd.read_csv(path))

    if per_class_all:
        per_class_all_df = pd.concat(per_class_all, ignore_index=True)
        per_class_all_df.to_csv(OUTPUT_DIR / "all_seed_per_class_metrics_selected_1se.csv", index=False)

        per_class_summary = (
            per_class_all_df.groupby(["split", "class_id", "class_name"])["f1"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "f1_mean",
                    "std": "f1_std",
                    "min": "f1_min",
                    "max": "f1_max",
                    "count": "num_seeds",
                }
            )
        )

        per_class_summary["f1_se"] = per_class_summary["f1_std"] / np.sqrt(
            per_class_summary["num_seeds"]
        )

        per_class_summary.to_csv(
            OUTPUT_DIR / "per_class_metrics_mean_std_se_selected_1se.csv",
            index=False,
        )

    per_env_all = []

    for seed in SEEDS:
        for split_name in ["ood_val", "ood_hold"]:
            path = OUTPUT_DIR / f"seed_{seed}" / f"{split_name}_per_environment_metrics_selected_1se.csv"
            if path.exists():
                per_env_all.append(pd.read_csv(path))

    if per_env_all:
        per_env_all_df = pd.concat(per_env_all, ignore_index=True)
        per_env_all_df.to_csv(OUTPUT_DIR / "all_seed_per_environment_metrics_selected_1se.csv", index=False)

        per_env_summary = (
            per_env_all_df.groupby(["split", "environment"])["macro_f1"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "macro_f1_mean",
                    "std": "macro_f1_std",
                    "min": "macro_f1_min",
                    "max": "macro_f1_max",
                    "count": "num_seeds",
                }
            )
        )

        per_env_summary["macro_f1_se"] = per_env_summary["macro_f1_std"] / np.sqrt(
            per_env_summary["num_seeds"]
        )

        per_env_summary.to_csv(
            OUTPUT_DIR / "per_environment_metrics_mean_std_se_selected_1se.csv",
            index=False,
        )

        hold_env_summary = per_env_summary[per_env_summary["split"] == "ood_hold"].copy()

        if len(hold_env_summary) > 0:
            worst_env_row = hold_env_summary.sort_values("macro_f1_mean").iloc[0].to_dict()
        else:
            worst_env_row = None
    else:
        worst_env_row = None

    final_summary = {
        "method": "resnet50_dro",
        "method_description": "DRO objective over disaster-location environments using train-based class-weighted cross entropy.",
        "seeds": SEEDS,
        "num_seeds": len(SEEDS),
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_workers": NUM_WORKERS,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "dro_eta": DRO_ETA,
        "dro_avg_risk_weight": DRO_AVG_RISK_WEIGHT,
        "dro_worst_risk_weight": DRO_WORST_RISK_WEIGHT,
        "device": str(device),
        "train_split": TRAIN_SPLIT,
        "validation_split": VAL_SPLIT,
        "final_test_split": FINAL_TEST_SPLIT,
        "environment_column": env_col,
        "training_groups": unique_train_groups,
        "unknown_group": UNKNOWN_GROUP,
        "unseen_validation_groups": unseen_val_groups,
        "unseen_hold_groups": unseen_hold_groups,
        "train_size": int(len(train_df)),
        "ood_validation_size": int(len(val_df)),
        "ood_hold_size": int(len(hold_df)),
        "model_selection": selection_info,
        "final_metric_summary": aggregate_df.to_dict(orient="records"),
        "worst_ood_hold_environment_by_mean_macro_f1": worst_env_row,
    }

    with open(OUTPUT_DIR / "final_summary_selected_1se.json", "w", encoding="utf-8") as f:
        json.dump(make_json_safe(final_summary), f, indent=2)

    print("\n" + "=" * 80)
    print("Final DRO OOD mean, std, and SE across seeds")
    print("=" * 80)
    print(aggregate_df)

    if worst_env_row is not None:
        print("\nWorst OOD hold environment by mean macro F1:")
        print(worst_env_row)

    print("\nSaved all DRO OOD outputs successfully.")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()