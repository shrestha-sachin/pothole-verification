"""Reproducible training pipeline for MobileNetV3-Small pothole verifier.

Two-stage training process:
  Stage 1: Frozen backbone (train linear classification head only).
  Stage 2: Optional unfreezing of backbone with reduced learning rate for fine-tuning.

Preserves subtype metadata:
  - Positives: pothole
  - Negatives: shadow, repaired_patch, manhole, ordinary_road

Enforces strict data hygiene:
  - Explicit split isolation (train vs. val only).
  - Rejects test, pilot, blank, and unknown splits from training.
  - Requires non-empty train and val sets with valid labels ('pothole' or 'non_pothole').
  - Validates image paths and readability; fails fast on unreadable images unless --allow-skip-invalid.
  - Refuses to fall back to full-frame image inventory.
  - Checks for source/group leakage between splits.
  - Computes validation loss, precision, recall, F1, and confusion matrix after each epoch.
  - Saves the best checkpoint using validation F1 score (breaking ties with validation loss).
  - Records seeds, hyperparameters, class mapping, and split metadata in the saved checkpoint.

DO NOT launch training automatically. This module provides the reproducible entry point.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    preprocess_crop_for_verifier,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLASS_NAMES = ["non_pothole", "pothole"]
CLASS_TO_IDX = {"non_pothole": 0, "pothole": 1}
SUBTYPES = ["shadow", "repaired_patch", "manhole", "ordinary_road", "pothole"]


class CorruptSampleError(Exception):
    """Raised when a crop image sample is unreadable or corrupt."""
    pass


class PotholeCropDataset(Dataset):
    """PyTorch Dataset loading and validating verified crops for verifier training."""

    def __init__(
        self,
        samples: Sequence[dict],
        target_size: int = TARGET_SIZE,
        is_training: bool = True,
        allow_skip_invalid: bool = False,
    ) -> None:
        self.target_size = target_size
        self.is_training = is_training
        self.allow_skip_invalid = allow_skip_invalid
        self.skipped_samples: list[dict] = []
        self.samples: list[dict] = []

        # Validate samples upfront before training begins
        for idx, item in enumerate(samples):
            crop_path = item.get("crop_filepath") or item.get("filepath")
            if not crop_path:
                if not allow_skip_invalid:
                    raise CorruptSampleError(
                        f"Sample at index {idx} (crop_id='{item.get('crop_id')}') has no crop_filepath."
                    )
                self.skipped_samples.append({**item, "reason": "missing_crop_filepath"})
                continue

            full_path = PROJECT_ROOT / crop_path if not Path(crop_path).is_absolute() else Path(crop_path)
            if not full_path.exists():
                if not allow_skip_invalid:
                    raise FileNotFoundError(
                        f"Crop image file does not exist: '{full_path}' (crop_id='{item.get('crop_id')}')."
                    )
                self.skipped_samples.append({**item, "reason": f"file_not_found: {full_path}"})
                continue

            # Verify readability
            bgr = cv2.imread(str(full_path))
            if bgr is None or bgr.size == 0:
                if not allow_skip_invalid:
                    raise CorruptSampleError(
                        f"Unreadable or corrupt image file: '{full_path}' (crop_id='{item.get('crop_id')}')."
                    )
                self.skipped_samples.append({**item, "reason": f"unreadable_image: {full_path}"})
                continue

            # Validate class label
            raw_label = item.get("label", "")
            if isinstance(raw_label, int) and raw_label in (0, 1):
                label_idx = raw_label
                label_str = CLASS_NAMES[label_idx]
            else:
                s_lbl = str(raw_label).strip().lower()
                if s_lbl in ("0", "1"):
                    label_idx = int(s_lbl)
                    label_str = CLASS_NAMES[label_idx]
                elif s_lbl in CLASS_TO_IDX:
                    label_idx = CLASS_TO_IDX[s_lbl]
                    label_str = s_lbl
                else:
                    raise ValueError(
                        f"Invalid class label '{raw_label}' for crop_id='{item.get('crop_id')}'. "
                        f"Must be one of {CLASS_NAMES} or 0/1."
                    )

            self.samples.append({
                **item,
                "_full_path": full_path,
                "_label_idx": label_idx,
                "_label_str": label_str,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        full_path = item["_full_path"]

        bgr = cv2.imread(str(full_path))
        if bgr is None or bgr.size == 0:
            raise CorruptSampleError(f"Failed to read image at runtime: '{full_path}'")

        # Standard preprocessing: aspect ratio preservation, 224x224 ImageNet mean pad, normalization
        tensor_arr = preprocess_crop_for_verifier(bgr, target_size=self.target_size)
        tensor = torch.from_numpy(tensor_arr)

        return {
            "image": tensor,
            "target": item["_label_idx"],
            "label": item["_label_str"],
            "subtype": item.get("subtype", ""),
            "crop_id": item.get("crop_id", str(idx)),
        }


def load_dataset_splits(
    inventory_path: Path,
    allow_skip_invalid: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load train, val, and excluded samples from crop_inventory.csv.

    Refuses to fall back to full-frame image inventories.
    Returns:
        (train_samples, val_samples, excluded_samples)
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"Crop inventory file not found: {inventory_path}")

    # Safety check: do not train verifier on raw full-frame image inventory
    if "crop_inventory" not in inventory_path.name.lower():
        raise ValueError(
            f"Refusing to train verifier on full-frame image inventory '{inventory_path.name}'. "
            "A valid crop inventory CSV (e.g. data/metadata/crop_inventory.csv) is required."
        )

    train_samples: list[dict] = []
    val_samples: list[dict] = []
    excluded_samples: list[dict] = []

    with inventory_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [fn.lower().strip() for fn in (reader.fieldnames or [])]
        if "crop_filepath" not in fieldnames and "filepath" not in fieldnames:
            raise ValueError(
                f"Invalid crop inventory format in {inventory_path}. Must contain 'crop_filepath' column."
            )

        for row_idx, row in enumerate(reader, start=2):
            status = row.get("review_status", "reviewed").strip().lower()
            uncertain = row.get("uncertain", "").strip().lower() in ("1", "true", "yes")
            label = row.get("label", "").strip().lower()
            split = row.get("split", "").strip().lower()

            item = {
                "crop_id": row.get("crop_id", str(row_idx)),
                "crop_filepath": row.get("crop_filepath") or row.get("filepath", ""),
                "label": label,
                "subtype": row.get("subtype", ""),
                "split": split,
                "source_image_id": row.get("source_image_id", ""),
                "recording_id": row.get("recording_id", ""),
                "video_id": row.get("video_id", ""),
                "sequence_id": row.get("sequence_id", ""),
            }

            # Exclude unreviewed, uncertain, test, pilot, or unknown splits
            if uncertain:
                excluded_samples.append({**item, "exclusion_reason": "marked_uncertain"})
                continue
            if status and status != "reviewed":
                excluded_samples.append({**item, "exclusion_reason": f"unreviewed_status: {status}"})
                continue

            if split in ("train", "training"):
                train_samples.append(item)
            elif split in ("val", "validation", "valid"):
                val_samples.append(item)
            else:
                # test, pilot, unassigned, blank are excluded from training by default
                excluded_samples.append({**item, "exclusion_reason": f"split_held_out: {split or 'blank'}"})

    return train_samples, val_samples, excluded_samples


def check_split_independence(
    train_samples: Sequence[dict],
    val_samples: Sequence[dict],
) -> dict[str, Any]:
    """Check for sequence/group overlap between train and validation splits to prevent data leakage."""
    report = {"checked": False, "overlap_detected": False, "overlapping_groups": [], "sufficient_metadata": True}

    def extract_groups(samples: Sequence[dict]) -> set[str]:
        groups = set()
        for s in samples:
            gid = (
                s.get("recording_id")
                or s.get("sequence_id")
                or s.get("video_id")
                or s.get("source_image_id")
            )
            if gid:
                groups.add(str(gid).strip())
        return groups

    train_groups = extract_groups(train_samples)
    val_groups = extract_groups(val_samples)

    if not train_groups or not val_groups:
        report["sufficient_metadata"] = False
        logger.warning(
            "Split independence check: Metadata (recording_id/video_id/sequence_id) is insufficient "
            "to strictly verify physical road-sequence independence between train and val."
        )
        return report

    report["checked"] = True
    overlap = sorted(train_groups.intersection(val_groups))
    if overlap:
        report["overlap_detected"] = True
        report["overlapping_groups"] = overlap
        raise ValueError(
            f"Data leakage detected! {len(overlap)} group/sequence IDs appear in BOTH train and val splits: {overlap}"
        )

    return report


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


def evaluate_verifier(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, Any]:
    """Calculate validation loss, accuracy, precision, recall, F1, and confusion matrix."""
    model.eval()
    total_loss = 0.0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            preds = torch.argmax(logits, dim=1)
            for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 0 and p == 0:
                    tn += 1
                elif t == 1 and p == 0:
                    fn += 1

    total_samples = tp + fp + tn + fn
    avg_loss = total_loss / max(1, total_samples)
    acc = (tp + tn) / max(1, total_samples)
    prec = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / max(1e-8, prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "val_accuracy": acc,
        "val_precision": prec,
        "val_recall": rec,
        "val_f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total_samples,
    }


def train_verifier(
    train_samples: list[dict],
    val_samples: list[dict],
    output_checkpoint: str | Path = "models/verifier_v1.pt",
    epochs_frozen: int = 5,
    epochs_finetune: int = 5,
    batch_size: int = 32,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    device: str = "auto",
    seed: int = 42,
    allow_skip_invalid: bool = False,
) -> Path:
    """Run reproducible two-stage training for MobileNetV3-Small with validation checkpoint selection."""
    if not train_samples:
        raise ValueError("Cannot train verifier: train_samples is empty.")
    if not val_samples:
        raise ValueError("Cannot train verifier: val_samples is empty. Held-out validation data is required.")

    # Seed reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    save_path = Path(output_checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() and device in ("auto", "cuda") else (
        "mps" if torch.backends.mps.is_available() and device in ("auto", "mps") else "cpu"
    )
    logger.info("Training MobileNetV3-Small on %s (Seed: %d)", dev, seed)

    # Check split independence
    check_split_independence(train_samples, val_samples)

    train_ds = PotholeCropDataset(train_samples, is_training=True, allow_skip_invalid=allow_skip_invalid)
    val_ds = PotholeCropDataset(val_samples, is_training=False, allow_skip_invalid=allow_skip_invalid)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = build_mobilenet_v3(pretrained=True, num_classes=2, device=dev)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_state_dict: Optional[dict] = None
    best_metrics: dict[str, Any] = {}

    # ── Stage 1: Freeze backbone, train classification head ──
    logger.info("Stage 1: Training classifier head with frozen backbone (%d epochs)...", epochs_frozen)
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_head)
    for ep in range(epochs_frozen):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(dev)
            targets = batch["target"].to(dev)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        train_loss = total_loss / max(1, len(train_ds))
        val_eval = evaluate_verifier(model, val_loader, criterion, dev)
        logger.info(
            "  Stage 1 Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f, Val F1: %.4f, Val Acc: %.4f",
            ep + 1, epochs_frozen, train_loss, val_eval["val_loss"], val_eval["val_f1"], val_eval["val_accuracy"]
        )

        # Checkpoint selection on validation F1 (tie-break on val loss)
        if (val_eval["val_f1"] > best_val_f1) or (val_eval["val_f1"] == best_val_f1 and val_eval["val_loss"] < best_val_loss):
            best_val_f1 = val_eval["val_f1"]
            best_val_loss = val_eval["val_loss"]
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {**val_eval, "best_epoch": ep + 1, "best_stage": 1}

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
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(dev)
                targets = batch["target"].to(dev)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(targets)

            train_loss = total_loss / max(1, len(train_ds))
            val_eval = evaluate_verifier(model, val_loader, criterion, dev)
            logger.info(
                "  Stage 2 Epoch %d/%d - Train Loss: %.4f | Val Loss: %.4f, Val F1: %.4f, Val Acc: %.4f",
                ep + 1, epochs_finetune, train_loss, val_eval["val_loss"], val_eval["val_f1"], val_eval["val_accuracy"]
            )

            if (val_eval["val_f1"] > best_val_f1) or (val_eval["val_f1"] == best_val_f1 and val_eval["val_loss"] < best_val_loss):
                best_val_f1 = val_eval["val_f1"]
                best_val_loss = val_eval["val_loss"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = {**val_eval, "best_epoch": ep + 1, "best_stage": 2}

    # Save checkpoint with full audit metadata
    checkpoint_data = {
        "model_state_dict": best_state_dict or model.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "classes": CLASS_NAMES,
        "subtypes": SUBTYPES,
        "selection_criterion": "highest_validation_f1_then_lowest_loss",
        "best_val_metrics": best_metrics,
        "training_settings": {
            "epochs_frozen": epochs_frozen,
            "epochs_finetune": epochs_finetune,
            "batch_size": batch_size,
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "seed": seed,
            "device": dev,
        },
        "split_identities": {
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "num_train_skipped": len(train_ds.skipped_samples),
            "num_val_skipped": len(val_ds.skipped_samples),
        },
        "preprocessing": {
            "target_size": TARGET_SIZE,
            "pad_frac": DEFAULT_PAD_FRAC,
            "mean_rgb": list(IMAGENET_MEAN_RGB),
            "std_rgb": list(IMAGENET_STD_RGB),
        },
    }
    torch.save(checkpoint_data, str(save_path))
    logger.info("Saved best verifier checkpoint to %s (Val F1: %.4f)", save_path, best_val_f1)
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--allow-skip-invalid", action="store_true",
                        help="Log and skip unreadable/missing crop files instead of failing")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and validate dataset splits without training")
    args = parser.parse_args()

    inv = args.inventory
    print(f"Loading verifier dataset splits from: {inv}")
    train_samples, val_samples, excluded = load_dataset_splits(inv, allow_skip_invalid=args.allow_skip_invalid)

    print("\n--- Verifier Dataset Split Summary ---")
    print(f"Train samples (split='train'):     {len(train_samples)}")
    print(f"Val samples (split='val'):         {len(val_samples)}")
    print(f"Excluded (test/pilot/uncertain):   {len(excluded)}")

    train_pos = sum(1 for s in train_samples if s["label"] == "pothole")
    train_neg = sum(1 for s in train_samples if s["label"] == "non_pothole")
    val_pos = sum(1 for s in val_samples if s["label"] == "pothole")
    val_neg = sum(1 for s in val_samples if s["label"] == "non_pothole")

    print(f"  Train: {train_pos} potholes, {train_neg} non-potholes")
    print(f"  Val:   {val_pos} potholes, {val_neg} non-potholes")

    # Independence check
    ind_report = check_split_independence(train_samples, val_samples)
    if ind_report.get("checked"):
        print("✓ Verified split independence: No overlapping sequence/group IDs.")

    if args.dry_run:
        print("\n[Dry-run completed successfully. Split hygiene and validation gates verified.]")
        return

    train_verifier(
        train_samples=train_samples,
        val_samples=val_samples,
        output_checkpoint=args.output,
        epochs_frozen=args.epochs_frozen,
        epochs_finetune=args.epochs_finetune,
        batch_size=args.batch_size,
        seed=args.seed,
        allow_skip_invalid=args.allow_skip_invalid,
    )


if __name__ == "__main__":
    main()
