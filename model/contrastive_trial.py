from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
)

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

OUTPUT_DIR = (
    BASE_DIR
    / "OOD_training_outputs"
    / "resnet50_supervised_contrastive_no_weights_test"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

BATCH_SIZE = 32
NUM_WORKERS = 2

CONTRASTIVE_EPOCHS = 4
CLASSIFIER_EPOCHS = 4

CONTRASTIVE_LR = 1e-4
CLASSIFIER_LR = 1e-4
ENCODER_FINETUNE_LR = 1e-5

LABEL_SMOOTHING = 0.10

TEMPERATURE = 0.10
PROJECTION_DIM = 128
FEATURE_DIM = 2048

TRAIN_SPLIT = "OOD_train"
VAL_SPLIT = "OOD_test"
HOLD_SPLIT = "OOD_hold"

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


set_seed(SEED)


# =========================
# Device
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
    def __init__(self, dataframe, contrastive=False):
        self.df = dataframe.reset_index(drop=True)
        self.contrastive = contrastive

    def __len__(self):
        return len(self.df)

    def load_crop(self, idx):
        row = self.df.iloc[idx]

        x = np.load(row["crop_path"])

        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        x = (
            x - IMAGENET_MEAN_6[:, None, None]
        ) / IMAGENET_STD_6[:, None, None]

        y = LABEL_TO_IDX[row["damage_label"]]

        return x, y

    def augment(self, x):
        x = x.copy()

        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()

        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()

        if random.random() < 0.25:
            noise = np.random.normal(
                0.0,
                0.01,
                size=x.shape,
            ).astype(np.float32)

            x = x + noise

        if random.random() < 0.25:
            scale = np.random.uniform(0.90, 1.10)
            shift = np.random.uniform(-0.03, 0.03)

            x = x * scale + shift

        return x.astype(np.float32)

    def __getitem__(self, idx):
        x, y = self.load_crop(idx)

        if self.contrastive:
            x1 = self.augment(x)
            x2 = self.augment(x)

            return (
                torch.tensor(x1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
            )

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


# =========================
# Model
# =========================

class ResNet50SixChannelEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2

        backbone = resnet50(weights=weights)

        old_conv = backbone.conv1

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

        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, projection_dim=128):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim),
        )

    def forward(self, features):
        z = self.projector(features)
        return F.normalize(z, dim=1)


class DamageClassifierHead(nn.Module):
    def __init__(self, input_dim=2048, num_classes=4):
        super().__init__()

        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


class FullDamageClassifier(nn.Module):
    def __init__(self, encoder, classifier_head):
        super().__init__()

        self.encoder = encoder
        self.classifier_head = classifier_head

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier_head(features)
        return logits


# =========================
# SupCon Loss
# =========================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.10):
        super().__init__()

        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        similarity = (
            torch.matmul(features, features.T)
            / self.temperature
        )

        logits_max, _ = torch.max(
            similarity,
            dim=1,
            keepdim=True,
        )

        logits = similarity - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        positives_mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True) + 1e-12
        )

        positives_per_sample = positives_mask.sum(dim=1)

        valid_mask = positives_per_sample > 0

        mean_log_prob_pos = (
            positives_mask * log_prob
        ).sum(dim=1) / (positives_per_sample + 1e-12)

        loss = -mean_log_prob_pos[valid_mask].mean()

        return loss


# =========================
# Evaluation
# =========================

def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()

    total_loss = 0.0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        progress_bar = tqdm(
            loader,
            desc=desc,
            leave=False,
        )

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

    df = df[df["damage_label"].isin(LABEL_TO_IDX.keys())].copy()

    train_df = df[df["split"] == TRAIN_SPLIT].copy()
    val_df = df[df["split"] == VAL_SPLIT].copy()
    hold_df = df[df["split"] == HOLD_SPLIT].copy()

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

    contrastive_loader = DataLoader(
        XViewBuildingDataset(
            train_df,
            contrastive=True,
        ),
        shuffle=True,
        **loader_kwargs,
    )

    train_loader = DataLoader(
        XViewBuildingDataset(
            train_df,
            contrastive=False,
        ),
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        XViewBuildingDataset(val_df),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df),
        shuffle=False,
        **loader_kwargs,
    )

    encoder = ResNet50SixChannelEncoder().to(device)

    projection_head = ProjectionHead(
        input_dim=FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(device)

    contrastive_criterion = SupConLoss(
        temperature=TEMPERATURE
    )

    contrastive_optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(projection_head.parameters()),
        lr=CONTRASTIVE_LR,
    )

    print("\nStarting supervised contrastive pretraining...")

    for epoch in range(1, CONTRASTIVE_EPOCHS + 1):
        encoder.train()
        projection_head.train()

        total_loss = 0.0

        epoch_start = time.time()

        progress_bar = tqdm(
            contrastive_loader,
            desc=f"Contrastive Epoch {epoch}/{CONTRASTIVE_EPOCHS}",
            leave=True,
        )

        for x1, x2, y in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            x = torch.cat([x1, x2], dim=0)
            y_all = torch.cat([y, y], dim=0)

            contrastive_optimizer.zero_grad()

            features = encoder(x)

            projections = projection_head(features)

            loss = contrastive_criterion(
                projections,
                y_all,
            )

            loss.backward()

            contrastive_optimizer.step()

            total_loss += loss.item() * x.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        avg_loss = (
            total_loss
            / (len(contrastive_loader.dataset) * 2)
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60.0

        print(
            f"Contrastive Epoch {epoch:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

    print(
        "\nUsing unweighted cross entropy with label smoothing:"
    )

    classifier_head = DamageClassifierHead(
        input_dim=FEATURE_DIM,
        num_classes=4,
    ).to(device)

    model = FullDamageClassifier(
        encoder=encoder,
        classifier_head=classifier_head,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_FINETUNE_LR,
            },
            {
                "params": model.classifier_head.parameters(),
                "lr": CLASSIFIER_LR,
            },
        ]
    )

    best_state = None
    best_f1 = -1.0

    print(
        "\nStarting classifier fine tuning "
        "without class weights..."
    )

    for epoch in range(1, CLASSIFIER_EPOCHS + 1):
        model.train()

        total_loss = 0.0

        epoch_start = time.time()

        progress_bar = tqdm(
            train_loader,
            desc=f"Classifier Epoch {epoch}/{CLASSIFIER_EPOCHS}",
            leave=True,
        )

        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item() * x.size(0)

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = total_loss / len(train_loader.dataset)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            desc=f"OOD validation eval epoch {epoch}",
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60.0

        print(
            f"Classifier Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"OOD Val Loss: {val_metrics['loss']:.4f} | "
            f"OOD Val Macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    print("\nEvaluating best model...")

    model.load_state_dict(best_state)

    final_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
        desc="Final OOD validation evaluation",
    )

    final_hold = evaluate(
        model,
        hold_loader,
        criterion,
        device,
        desc="Final OOD hold evaluation",
    )

    print(
        f"\nFinal OOD VAL Macro F1: "
        f"{final_val['macro_f1']:.4f}"
    )

    print(
        classification_report(
            final_val["targets"],
            final_val["preds"],
            labels=LABEL_IDS,
            target_names=[
                IDX_TO_LABEL[i]
                for i in LABEL_IDS
            ],
            digits=4,
            zero_division=0,
        )
    )

    print(
        f"\nFinal OOD HOLD Macro F1: "
        f"{final_hold['macro_f1']:.4f}"
    )

    print(
        classification_report(
            final_hold["targets"],
            final_hold["preds"],
            labels=LABEL_IDS,
            target_names=[
                IDX_TO_LABEL[i]
                for i in LABEL_IDS
            ],
            digits=4,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()