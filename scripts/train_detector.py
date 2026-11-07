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
import json
from datetime import datetime, timezone
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
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Unique run name; defaults to a UTC timestamp")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow overwriting an existing run and checkpoint")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate an existing --model checkpoint without training or modifying weights")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Split to use with --evaluate-only (default: test)")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate final model on protected held-out test split after training (never during tuning)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset configuration and paths without launching training",
    )

    args = parser.parse_args(argv)
    if args.evaluate_only and args.evaluate_test:
        parser.error("--evaluate-only and --evaluate-test cannot be combined")
    if args.evaluate_only and not Path(args.model).is_file():
        parser.error("--evaluate-only requires --model pointing to an existing checkpoint")

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
    if data_dict.get("test"):
        print(f"  Test sources:  {data_dict.get('test')}")

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
    n_test = count_images_in_sources(data_dict.get("test", []))
    print(f"  Total train images: {n_train}")
    print(f"  Total val images:   {n_val}")
    print(f"  Total test images:  {n_test} (held-out, protected)")

    if args.dry_run:
        print("\n[Dry-run passed successfully. Dataset YAML parsed; inspect reported counts before training.]")
        return

    if args.evaluate_only:
        if args.split == "test" and (not data_dict.get("test") or n_test == 0):
            sys.exit("ERROR: Held-out test split is unavailable or empty.")
        if args.split == "val" and n_val == 0:
            sys.exit("ERROR: Validation split is empty.")
        print(f"\n[Evaluation only] Loading {args.model} for {args.split} evaluation; no training.")
        metrics = YOLO(args.model).val(data=str(data_yaml_path), split=args.split, device=device)
        if hasattr(metrics, "box"):
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall:    {metrics.box.mr:.4f}")
            print(f"  mAP50:     {metrics.box.map50:.4f}")
            print(f"  mAP50-95:  {metrics.box.map:.4f}")
        return

    run_name = args.experiment_name or datetime.now(timezone.utc).strftime("pothole_%Y%m%dT%H%M%S_%fZ")
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--experiment-name must be a single directory name")
    run_dir_target = PROJECT_ROOT / "runs" / "detect" / run_name
    if run_dir_target.exists() and not args.overwrite:
        sys.exit(f"ERROR: Run already exists: {run_dir_target}. Choose another --experiment-name.")
    if args.output.exists() and not args.overwrite:
        sys.exit(f"ERROR: Checkpoint already exists: {args.output}. Choose a versioned --output, or explicitly pass --overwrite.")
    if args.evaluate_test and (not data_dict.get("test") or n_test == 0):
        sys.exit("ERROR: --evaluate-test requested but held-out test split is missing or empty.")

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
        name=run_name,
        exist_ok=args.overwrite,
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

    if args.evaluate_test and n_test > 0:
        print("\n[Evaluation] Evaluating best model on held-out test set...")
        test_metrics = model.val(data=str(data_yaml_path), split="test", device=device)
        print("\n--- Test Metrics (Final Benchmark) ---")
        if hasattr(test_metrics, "box"):
            print(f"  Precision: {test_metrics.box.mp:.4f}")
            print(f"  Recall:    {test_metrics.box.mr:.4f}")
            print(f"  mAP50:     {test_metrics.box.map50:.4f}")
            print(f"  mAP50-95:  {test_metrics.box.map:.4f}")

    # 6. Copy best weights to models/best.pt
    run_dir = Path(model.trainer.save_dir)
    best_weights_path = run_dir / "weights" / "best.pt"

    if best_weights_path.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidental overwrite even if a second process
        # created the checkpoint after the preflight check.
        with best_weights_path.open("rb") as source, args.output.open("wb" if args.overwrite else "xb") as destination:
            shutil.copyfileobj(source, destination)
        print(f"\n[OK] Copied best model weights to: {args.output}")
        print("  The Streamlit dashboard may use this checkpoint when configured to do so.")
    else:
        print(f"\n[WARN] Best weights file not found at expected location: {best_weights_path}")


if __name__ == "__main__":
    main()

