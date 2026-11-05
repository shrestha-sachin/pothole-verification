"""Reproducible training and evaluation script for YOLO11n pothole detector.

Trains on the combined dataset (Local Green Bay smartphone frames + Mapillary
segments + balanced RDD2022 US dataset) and validates on held-out test splits.
Automatically exports best weights to models/best.pt for the Streamlit dashboard.

Usage:
    # Dry-run validation of dataset configuration
    python scripts/train_detector.py --dry-run

    # Train on GPU (if CUDA available, auto-detected) or CPU
    python scripts/train_detector.py --epochs 50 --batch 16

    # Custom model base or device
    python scripts/train_detector.py --model yolo11n.pt --device 0 --epochs 100
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = PROJECT_ROOT / "configs" / "detector.yaml"
DEFAULT_BASE_MODEL = PROJECT_ROOT / "yolo11n.pt"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "models" / "best.pt"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate YOLO pothole detector on combined dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_BASE_MODEL if DEFAULT_BASE_MODEL.exists() else "yolo11n.pt"),
        help="Base YOLO model weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="YOLO dataset YAML config (default: configs/detector.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes (default: 2 for fast, freeze-free startup on Windows)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device (e.g. '0', 'cpu', or empty for auto-detection)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help="Destination path for best trained model checkpoint (default: models/best.pt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset configuration and paths without launching training",
    )

    args = parser.parse_args(argv)

    # 1. Device selection
    if args.device:
        device = args.device
    else:
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using compute device: {device} (CUDA available: {torch.cuda.is_available()})")

    # 2. Check dataset configuration
    data_yaml_path = args.data.resolve()
    if not data_yaml_path.exists():
        sys.exit(f"ERROR: Dataset YAML config does not exist at {data_yaml_path}")

    print(f"[Dataset] Validating YAML configuration: {data_yaml_path}")
    data_dict = check_det_dataset(str(data_yaml_path))
    print(f"  Classes: {data_dict.get('names')}")
    print(f"  Train sources: {data_dict.get('train')}")
    print(f"  Val sources:   {data_dict.get('val')}")

    # Count total images in splits
    def count_images_in_sources(sources) -> int:
        if isinstance(sources, (str, Path)):
            sources = [sources]
        total = 0
        for s in sources:
            p = PROJECT_ROOT / s if not Path(s).is_absolute() else Path(s)
            if p.exists() and p.is_file():
                total += sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())
        return total

    n_train = count_images_in_sources(data_dict.get("train", []))
    n_val = count_images_in_sources(data_dict.get("val", []))
    print(f"  Total train images: {n_train}")
    print(f"  Total val images:   {n_val}")

    if args.dry_run:
        print("\n[Dry-run passed successfully. All dataset manifests and labels resolve correctly.]")
        return

    # 3. Load base model
    print(f"\n[Model] Loading base weights: {args.model}")
    model = YOLO(args.model)

    # 4. Train
    print(f"\n[Training] Starting training for {args.epochs} epochs (batch={args.batch}, imgsz={args.imgsz}, workers={args.workers})...", flush=True)
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="pothole_combined",
        exist_ok=True,
    )

    # 5. Evaluate on held-out validation set
    print("\n[Evaluation] Evaluating best model on validation set...")
    val_metrics = model.val(data=str(data_yaml_path), split="val", device=device)

    print("\n--- Validation Metrics ---")
    if hasattr(val_metrics, "box"):
        print(f"  Precision: {val_metrics.box.mp:.4f}")
        print(f"  Recall:    {val_metrics.box.mr:.4f}")
        print(f"  mAP50:     {val_metrics.box.map50:.4f}")
        print(f"  mAP50-95:  {val_metrics.box.map:.4f}")

    # 6. Copy best weights to models/best.pt
    run_dir = Path(model.trainer.save_dir)
    best_weights_path = run_dir / "weights" / "best.pt"

    if best_weights_path.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights_path, args.output)
        print(f"\n[OK] Copied best model weights to: {args.output}")
        print("  The Streamlit dashboard (dashboard/app.py) will now use this checkpoint!")
    else:
        print(f"\n[WARN] Best weights file not found at expected location: {best_weights_path}")


if __name__ == "__main__":
    main()

