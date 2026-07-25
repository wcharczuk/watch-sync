#!/usr/bin/env python3
"""Train AngleNet: MobileNetV2 backbone for watch second hand angle detection.

Uses pretrained ImageNet features for robust edge/line detection,
with a custom regression head for (sin, cos) angle output.

Usage:
    python train.py                  # Train from scratch on synthetic data
    python train.py --finetune       # Fine-tune on synthetic + real data
    python train.py --resume         # Resume from checkpoint
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BATCH_SIZE = 64
NUM_EPOCHS = 20
FINETUNE_EPOCHS = 10
NUM_WORKERS = 0


def angular_loss(pred, target):
    """Loss that directly penalizes angular error and encourages unit-norm output.

    pred, target: (B, 2) tensors of (sin, cos) values.
    """
    pred_norm = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
    cos_diff = (pred_norm * target).sum(dim=1).clamp(-1, 1)
    return (1 - cos_diff).mean()


class AngleNet(nn.Module):
    """MobileNetV2 backbone with angle regression head.

    Uses pretrained features for robust line/edge detection.
    Output: (sin_theta, cos_theta) for the second hand angle.
    """

    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # (sin_theta, cos_theta)
        )
        # Initialize head
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Break the (0,0) saddle point
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([0.0, 1.0]))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class WatchDataset(Dataset):
    """Dataset loading watch face images and (sin, cos) labels from CSV."""

    def __init__(self, img_dir, csv_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "sin_theta": float(row["sin_theta"]),
                    "cos_theta": float(row["cos_theta"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.img_dir, sample["filename"])
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        target = torch.tensor([sample["sin_theta"], sample["cos_theta"]], dtype=torch.float32)
        return img, target


def angular_error_degrees(pred_sin, pred_cos, gt_sin, gt_cos):
    """Compute mean absolute angular error in degrees."""
    pred_angle = torch.atan2(pred_sin, pred_cos)
    gt_angle = torch.atan2(gt_sin, gt_cos)
    diff = pred_angle - gt_angle
    # Wrap to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff).mean() * 180.0 / math.pi


def build_datasets(finetune=False):
    """Build train and val datasets, optionally including real data.

    Returns (train_dataset, val_dataset, sampler, real_val_dataset).
    real_val_dataset is a separate dataset for tracking real-data accuracy.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        normalize,
    ])

    # Stronger augmentation for real data (fewer samples, need more variety)
    real_train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.03),
        transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # Synthetic datasets
    train_dataset = WatchDataset(
        os.path.join(DATA_DIR, "train"),
        os.path.join(DATA_DIR, "train_labels.csv"),
        transform=train_transform,
    )
    val_dataset = WatchDataset(
        os.path.join(DATA_DIR, "val"),
        os.path.join(DATA_DIR, "val_labels.csv"),
        transform=val_transform,
    )

    sampler = None
    real_val_dataset = None

    # Prefer v4-labeled data, fall back to bootstrap-labeled
    v4_train_csv = os.path.join(DATA_DIR, "real_v4_train_labels.csv")
    v4_val_csv = os.path.join(DATA_DIR, "real_v4_val_labels.csv")
    old_train_csv = os.path.join(DATA_DIR, "real_train_labels.csv")
    old_val_csv = os.path.join(DATA_DIR, "real_val_labels.csv")

    if os.path.exists(v4_train_csv):
        real_train_csv = v4_train_csv
        real_train_dir = os.path.join(DATA_DIR, "real_v4_train")
    elif os.path.exists(old_train_csv):
        real_train_csv = old_train_csv
        real_train_dir = os.path.join(DATA_DIR, "real_train")
    else:
        real_train_csv = None
        real_train_dir = None

    if os.path.exists(v4_val_csv):
        real_val_csv = v4_val_csv
        real_val_dir = os.path.join(DATA_DIR, "real_v4_val")
    elif os.path.exists(old_val_csv):
        real_val_csv = old_val_csv
        real_val_dir = os.path.join(DATA_DIR, "real_val")
    else:
        real_val_csv = None
        real_val_dir = None

    # Always load real val dataset for separate tracking (if available)
    if real_val_csv and real_val_dir:
        real_val_dataset = WatchDataset(real_val_dir, real_val_csv, transform=val_transform)
        print(f"  Real val samples: {len(real_val_dataset)} (from {os.path.basename(real_val_csv)})")

    if finetune:
        if real_train_csv and real_train_dir:
            real_train = WatchDataset(real_train_dir, real_train_csv,
                                      transform=real_train_transform)
            print(f"  Real train samples: {len(real_train)} (from {os.path.basename(real_train_csv)})")

            n_synthetic = len(train_dataset)
            n_real = len(real_train)

            combined = ConcatDataset([train_dataset, real_train])

            # Target: ~30% of each batch is real data
            real_weight = (0.3 * n_synthetic) / (0.7 * n_real) if n_real > 0 else 1.0
            weights = [1.0] * n_synthetic + [real_weight] * n_real
            sampler = WeightedRandomSampler(
                weights, num_samples=len(combined), replacement=True,
            )
            train_dataset = combined
            print(f"  Combined dataset: {len(combined)} (real weight: {real_weight:.1f}x)")

        # Combine synthetic + real val for overall tracking
        if real_val_dataset:
            val_dataset = ConcatDataset([val_dataset, real_val_dataset])

    return train_dataset, val_dataset, sampler, real_val_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune with mixed synthetic + real data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from best checkpoint")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = FINETUNE_EPOCHS if args.finetune else NUM_EPOCHS

    print("Building datasets...")
    train_dataset, val_dataset, sampler, real_val_dataset = build_datasets(finetune=args.finetune)
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=(sampler is None),  # Don't shuffle if using sampler
        sampler=sampler,
        num_workers=NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )
    real_val_loader = None
    if real_val_dataset and len(real_val_dataset) > 0:
        real_val_loader = DataLoader(
            real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS,
        )

    model = AngleNet().to(device)

    # Load checkpoint if resuming or fine-tuning
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if (args.resume or args.finetune) and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = angular_loss

    # Different LR for backbone (lower) and head (higher)
    # Use lower LR when fine-tuning
    backbone_lr = 3e-5 if args.finetune else 1e-4
    head_lr = 3e-4 if args.finetune else 1e-3

    backbone_params = list(model.features.parameters())
    head_params = list(model.head.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_error = float("inf")
    epoch_times = []
    training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_angle_error = 0.0
        num_train_batches = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                ae = angular_error_degrees(
                    outputs[:, 0], outputs[:, 1],
                    targets[:, 0], targets[:, 1],
                )
                train_angle_error += ae.item()
            num_train_batches += 1

        scheduler.step()

        avg_train_loss = train_loss / num_train_batches
        avg_train_ae = train_angle_error / num_train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_angle_error = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                ae = angular_error_degrees(
                    outputs[:, 0], outputs[:, 1],
                    targets[:, 0], targets[:, 1],
                )
                val_angle_error += ae.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        avg_val_ae = val_angle_error / num_val_batches

        # Separate real-data validation
        real_val_ae_str = ""
        if real_val_loader:
            real_val_error = 0.0
            num_real_batches = 0
            with torch.no_grad():
                for images, targets in real_val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    outputs = model(images)
                    ae = angular_error_degrees(
                        outputs[:, 0], outputs[:, 1],
                        targets[:, 0], targets[:, 1],
                    )
                    real_val_error += ae.item()
                    num_real_batches += 1
            avg_real_ae = real_val_error / max(num_real_batches, 1)
            real_val_ae_str = f" | Real AE: {avg_real_ae:.2f}°"

        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = f"{int(eta_seconds // 60)}m{int(eta_seconds % 60):02d}s" if remaining_epochs > 0 else "done"

        mode = "FT" if args.finetune else "  "
        print(
            f"{mode} Epoch {epoch + 1:2d}/{num_epochs} "
            f"[{epoch_elapsed:.0f}s, ETA {eta_str}] | "
            f"Train Loss: {avg_train_loss:.4f} AE: {avg_train_ae:.2f}° | "
            f"Val Loss: {avg_val_loss:.4f} AE: {avg_val_ae:.2f}°"
            f"{real_val_ae_str}"
        )

        if avg_val_ae < best_val_error:
            best_val_error = avg_val_ae
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  -> Saved best model (val AE: {avg_val_ae:.2f}°)")

    total_time = time.time() - training_start
    print(f"\nTraining complete in {int(total_time // 60)}m{int(total_time % 60):02d}s. "
          f"Best validation angular error: {best_val_error:.2f}°")


if __name__ == "__main__":
    main()
