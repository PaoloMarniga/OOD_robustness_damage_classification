from pathlib import Path
import random
import json
import copy
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights


BASE_DIR = Path.home() / "Desktop"
CSV_PATH = BASE_DIR / "OOD_processed" / "buildings_all_OOD_with_crops.csv"

OUTPUT_DIR = BASE_DIR / "OOD_training_outputs" / "resnet50_supervised_contrastive"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 2

CONTRASTIVE_EPOCHS = 5
CLASSIFIER_EPOCHS = 5

CONTRASTIVE_LR = 1e-4
CLASSIFIER_LR = 1e-4
ENCODER_FINETUNE_LR = 1e-5

LABEL_SMOOTHING = 0.10

TEMPERATURE = 0.10
PROJECTION_DIM = 128
FEATURE_DIM = 2048

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class XViewBuildingDataset(Dataset):
    def __init__(self, dataframe, train=False, contrastive=False):
        self.df = dataframe.reset_index(drop=True)
        self.train = train
        self.contrastive = contrastive

    def __len__(self):
        return len(self.df)

    def load_crop(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["crop_path"])
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        y = LABEL_TO_IDX[row["damage_label"]]
        return x, y

    def augment(self, x):
        x = x.copy()

        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()

        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()

        if random.random() < 0.25:
            noise = np.random.normal(0.0, 0.01, size=x.shape).astype(np.float32)
            x = np.clip(x + noise, 0.0, 1.0)

        if random.random() < 0.25:
            scale = np.random.uniform(0.90, 1.10)
            shift = np.random.uniform(-0.03, 0.03)
            x = np.clip(x * scale + shift, 0.0, 1.0)

        return x

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

        if self.train:
            x = self.augment(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


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
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight

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


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.10):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        similarity = torch.matmul(features, features.T) / self.temperature

        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        positives_mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_sample = positives_mask.sum(dim=1)
        valid_mask = positives_per_sample > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (
            positives_mask * log_prob
        ).sum(dim=1) / (positives_per_sample + 1e-12)

        return -mean_log_prob_pos[valid_mask].mean()


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

    pred_counts = pd.Series(preds_all).value_counts().sort_index().to_dict()
    target_counts = pd.Series(targets_all).value_counts().sort_index().to_dict()

    return {
        "loss": loss_avg,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "preds": preds_all,
        "targets": targets_all,
        "pred_counts": pred_counts,
        "target_counts": target_counts,
    }


def print_distribution(metrics, title):
    print(f"\n{title} prediction distribution:")
    for idx in range(4):
        print(f"{IDX_TO_LABEL[idx]}: {metrics['pred_counts'].get(idx, 0)}")

    print(f"\n{title} true distribution:")
    for idx in range(4):
        print(f"{IDX_TO_LABEL[idx]}: {metrics['target_counts'].get(idx, 0)}")


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

    train_locations = set(train_df["disaster"])
    test_locations = set(test_df["disaster"])
    hold_locations = set(hold_df["disaster"])

    train_ids = set(train_df["image_id"])
    test_ids = set(test_df["image_id"])
    hold_ids = set(hold_df["image_id"])

    print("\nImage overlap check:")
    print("OOD_train intersection OOD_test:", len(train_ids & test_ids))
    print("OOD_train intersection OOD_hold:", len(train_ids & hold_ids))
    print("OOD_test intersection OOD_hold:", len(test_ids & hold_ids))

    print("\nLocation overlap check:")
    print("OOD_train intersection OOD_test:", len(train_locations & test_locations))
    print("OOD_train intersection OOD_hold:", len(train_locations & hold_locations))
    print("OOD_test intersection OOD_hold:", len(test_locations & hold_locations))

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

    contrastive_loader = DataLoader(
        XViewBuildingDataset(train_df, train=True, contrastive=True),
        shuffle=True,
        **loader_kwargs,
    )

    train_loader = DataLoader(
        XViewBuildingDataset(train_df, train=True, contrastive=False),
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        XViewBuildingDataset(test_df, train=False, contrastive=False),
        shuffle=False,
        **loader_kwargs,
    )

    hold_loader = DataLoader(
        XViewBuildingDataset(hold_df, train=False, contrastive=False),
        shuffle=False,
        **loader_kwargs,
    )

    device = get_device()
    print(f"\nUsing device: {device}")

    encoder = ResNet50SixChannelEncoder().to(device)

    projection_head = ProjectionHead(
        input_dim=FEATURE_DIM,
        projection_dim=PROJECTION_DIM,
    ).to(device)

    contrastive_criterion = SupConLoss(temperature=TEMPERATURE)

    contrastive_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projection_head.parameters()),
        lr=CONTRASTIVE_LR,
    )

    contrastive_history = []

    print("\nStarting ResNet50 supervised contrastive pretraining...")

    for epoch in range(CONTRASTIVE_EPOCHS):
        epoch_start = time.time()

        encoder.train()
        projection_head.train()

        total_loss = 0.0

        progress_bar = tqdm(
            contrastive_loader,
            desc=f"Contrastive Epoch {epoch + 1}/{CONTRASTIVE_EPOCHS}",
            leave=True,
        )

        for x1, x2, y in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            x = torch.cat([x1, x2], dim=0)
            y_contrastive = torch.cat([y, y], dim=0)

            contrastive_optimizer.zero_grad()

            features = encoder(x)
            projections = projection_head(features)

            loss = contrastive_criterion(projections, y_contrastive)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during contrastive training.")

            loss.backward()
            contrastive_optimizer.step()

            total_loss += loss.item() * x.size(0)

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_minutes = (time.time() - epoch_start) / 60.0
        avg_loss = total_loss / (len(contrastive_loader.dataset) * 2)

        contrastive_history.append(
            {
                "epoch": epoch + 1,
                "contrastive_loss": avg_loss,
                "epoch_minutes": epoch_minutes,
            }
        )

        print(
            f"Contrastive Epoch {epoch + 1:02d} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {epoch_minutes:.2f} min"
        )

        torch.save(
            {
                "encoder": encoder.state_dict(),
                "projection_head": projection_head.state_dict(),
                "feature_dim": FEATURE_DIM,
                "projection_dim": PROJECTION_DIM,
            },
            OUTPUT_DIR / "contrastive_latest.pt",
        )

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "projection_head": projection_head.state_dict(),
            "feature_dim": FEATURE_DIM,
            "projection_dim": PROJECTION_DIM,
        },
        OUTPUT_DIR / "contrastive_final.pt",
    )

    pd.DataFrame(contrastive_history).to_csv(
        OUTPUT_DIR / "contrastive_history.csv",
        index=False,
    )

    print("\nStarting classifier training on contrastive ResNet50 encoder...")

    classifier_head = DamageClassifierHead(
        input_dim=FEATURE_DIM,
        num_classes=4,
    ).to(device)

    model = FullDamageClassifier(
        encoder=encoder,
        classifier_head=classifier_head,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    print(
        "\nUsing cross entropy loss without class weights "
        f"and with label_smoothing={LABEL_SMOOTHING}"
    )

    classifier_optimizer = torch.optim.Adam(
        [
            {"params": model.encoder.parameters(), "lr": ENCODER_FINETUNE_LR},
            {"params": model.classifier_head.parameters(), "lr": CLASSIFIER_LR},
        ]
    )

    best_state = None
    best_f1 = -1.0
    classifier_history = []

    for epoch in range(CLASSIFIER_EPOCHS):
        epoch_start = time.time()

        model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Classifier Epoch {epoch + 1}/{CLASSIFIER_EPOCHS}",
            leave=True,
        )

        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            classifier_optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss detected during classifier training.")

            loss.backward()
            classifier_optimizer.step()

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

        classifier_history.append(
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
            f"Classifier Epoch {epoch + 1:02d} | "
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

    pd.DataFrame(classifier_history).to_csv(
        OUTPUT_DIR / "classifier_history.csv",
        index=False,
    )

    print("\nEvaluating best ResNet50 supervised contrastive model...")
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

    print_distribution(final_test, "OOD TEST")

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

    print_distribution(final_hold, "OOD HOLD")

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
        "contrastive_epochs": CONTRASTIVE_EPOCHS,
        "classifier_epochs": CLASSIFIER_EPOCHS,
        "total_epochs": CONTRASTIVE_EPOCHS + CLASSIFIER_EPOCHS,
        "contrastive_learning_rate": CONTRASTIVE_LR,
        "classifier_learning_rate": CLASSIFIER_LR,
        "encoder_finetune_learning_rate": ENCODER_FINETUNE_LR,
        "temperature": TEMPERATURE,
        "projection_dim": PROJECTION_DIM,
        "feature_dim": FEATURE_DIM,
        "num_workers": NUM_WORKERS,
        "use_class_weights": False,
        "label_smoothing": LABEL_SMOOTHING,
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
        "method": "ResNet50 six-channel encoder with supervised contrastive representation pretraining followed by classifier training without class weights.",
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(OUTPUT_DIR / "ood_test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)

    with open(OUTPUT_DIR / "ood_hold_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(hold_report, f, indent=2)

    print("\nSaved all ResNet50 supervised contrastive outputs successfully.")
    print(OUTPUT_DIR / "best_model.pt")
    print(OUTPUT_DIR / "classifier_history.csv")
    print(OUTPUT_DIR / "summary.json")


if __name__ == "__main__":
    main()