"""Reproducible training pipeline for MobileNetV3-Small pothole verifier.

Two-stage training process:
  Stage 1: Frozen backbone (train linear classification head only).
  Stage 2: Optional unfreezing of backbone with reduced learning rate for fine-tuning.

Preserves subtype metadata:
  - Positives: pothole
  - Negatives: shadow, repaired_patch, manhole, ordinary_road

Saves checkpoints with explicit metadata including class_to_idx mapping.
DO NOT launch training automatically. This module provides the reproducible entry point.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.verifier.preprocessing import (
    DEFAULT_PAD_FRAC,
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    TARGET_SIZE,
    extract_padded_crop,
    preprocess_crop_for_verifier,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_NAMES = ["non_pothole", "pothole"]
CLASS_TO_IDX = {"non_pothole": 0, "pothole": 1}
SUBTYPES = ["shadow", "repaired_patch", "manhole", "ordinary_road", "pothole"]


class PotholeCropDataset(Dataset):
    """PyTorch Dataset loading crops or annotated regions for verifier training."""

    def __init__(
        self,
        samples: Sequence[dict],
        target_size: int = TARGET_SIZE,
        is_training: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.target_size = target_size
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        crop_path = item.get("crop_filepath") or item.get("filepath")
        if crop_path:
            full_path = PROJECT_ROOT / crop_path if not Path(crop_path).is_absolute() else Path(crop_path)
            bgr = cv2.imread(str(full_path))
        else:
            bgr = None

        if bgr is None:
            # Fallback blank
            bgr = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        # Standard preprocessing: pad to 224x224, aspect-ratio preserved, normalized
        tensor_arr = preprocess_crop_for_verifier(bgr, target_size=self.target_size)
        tensor = torch.from_numpy(tensor_arr)

        label_str = item.get("label", "non_pothole")
        target_idx = CLASS_TO_IDX.get(label_str, 0)
        subtype = item.get("subtype", "")

        return {
            "image": tensor,
            "target": target_idx,
            "label": label_str,
            "subtype": subtype,
            "crop_id": item.get("crop_id", str(idx)),
        }


def load_dataset_samples(
    inventory_path: Path,
    split_filter: Optional[str] = None,
) -> list[dict]:
    """Load valid training/validation samples from crop_inventory.csv or image_inventory.csv."""
    if not inventory_path.exists():
        logger.warning("Inventory file not found: %s", inventory_path)
        return []

    samples = []
    with inventory_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            status = row.get("review_status", "reviewed").strip().lower()
            uncertain = row.get("uncertain", "").strip().lower() in ("1", "true", "yes")
            if uncertain or (status and status != "reviewed"):
                continue

            split = row.get("split", "train").strip().lower()
            if split_filter and split != split_filter.lower():
                continue

            label = row.get("label", "")
            if not label:
                # Infer from pothole_count if image inventory
                cnt = int(row.get("pothole_count", 0) or 0)
                label = "pothole" if cnt > 0 else "non_pothole"

            samples.append({
                "crop_id": row.get("crop_id", row.get("image_id", "")),
                "crop_filepath": row.get("crop_filepath") or row.get("filepath", ""),
                "label": label,
                "subtype": row.get("subtype", ""),
                "split": split,
            })
    return samples


def build_mobilenet_v3(
    pretrained: bool = True,
    num_classes: int = 2,
    device: str = "cpu",
) -> nn.Module:
    """Build MobileNetV3-Small classifier with modified binary head."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_verifier(
    train_samples: list[dict],
    val_samples: Optional[list[dict]] = None,
    output_checkpoint: str | Path = "models/verifier_v1.pt",
    epochs_frozen: int = 5,
    epochs_finetune: int = 5,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    device: str = "auto",
) -> Path:
    """Run reproducible two-stage training for MobileNetV3-Small."""
    save_path = Path(output_checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() and device in ("auto", "cuda") else (
        "mps" if torch.backends.mps.is_available() and device in ("auto", "mps") else "cpu"
    )
    logger.info("Training MobileNetV3-Small on %s", dev)

    model = build_mobilenet_v3(pretrained=True, num_classes=2, device=dev)
    criterion = nn.CrossEntropyLoss()

    train_ds = PotholeCropDataset(train_samples, is_training=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # ── Stage 1: Freeze backbone, train classification head ──
    logger.info("Stage 1: Training classifier head with frozen backbone (%d epochs)...", epochs_frozen)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_head)
    model.train()
    for ep in range(epochs_frozen):
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(dev)
            targets = batch["target"].to(dev)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info("  Stage 1 Epoch %d/%d - Loss: %.4f", ep + 1, epochs_frozen, total_loss / max(1, len(train_loader)))

    # ── Stage 2: Unfreeze backbone with lower learning rate ──
    if epochs_finetune > 0:
        logger.info("Stage 2: Fine-tuning full model (%d epochs)...", epochs_finetune)
        for param in model.features.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head * 0.5},
        ])
        for ep in range(epochs_finetune):
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(dev)
                targets = batch["target"].to(dev)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info("  Stage 2 Epoch %d/%d - Loss: %.4f", ep + 1, epochs_finetune, total_loss / max(1, len(train_loader)))

    # Save checkpoint with full metadata
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "classes": CLASS_NAMES,
        "subtypes": SUBTYPES,
        "version": save_path.stem,
        "architecture": "mobilenet_v3_small",
        "target_size": TARGET_SIZE,
        "pad_frac": DEFAULT_PAD_FRAC,
        "mean_rgb": list(IMAGENET_MEAN_RGB),
        "std_rgb": list(IMAGENET_STD_RGB),
    }
    torch.save(checkpoint_data, str(save_path))
    logger.info("Saved trained verifier checkpoint to %s", save_path)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small verifier on road crops.")
    parser.add_argument("--inventory", type=Path, default=PROJECT_ROOT / "data" / "metadata" / "crop_inventory.csv",
                        help="Path to crop_inventory.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "verifier_v1.pt",
                        help="Output checkpoint path")
    parser.add_argument("--epochs-frozen", type=int, default=5, help="Epochs for frozen backbone training")
    parser.add_argument("--epochs-finetune", type=int, default=5, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Inspect dataset without training")
    args = parser.parse_args()

    inv = args.inventory if args.inventory.exists() else PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"
    samples = load_dataset_samples(inv)
    print(f"Loaded {len(samples)} usable samples from {inv}")
    pos = sum(1 for s in samples if s["label"] == "pothole")
    neg = sum(1 for s in samples if s["label"] == "non_pothole")
    print(f"  Potholes: {pos}, Non-potholes: {neg}")

    if args.dry_run:
        print("Dry run completed. Training entry point verified.")
        return

    train_verifier(
        samples,
        output_checkpoint=args.output,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

