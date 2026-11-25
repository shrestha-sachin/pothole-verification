"""Reproducible training and evaluation script for YOLO11 / YOLO26 pothole detectors.

Reads defaults from configs/training.yaml and configs/detector.yaml.
Forces explicit optimizer/lr0 hyperparameters, verifies clean base-weight
initialization, saves frozen manifest provenance in .meta.json, and validates
dataset integrity before training; test evaluation is strictly opt-in.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import sha256_file
from src.utils.config import (
    RUNS_DIR,
    SPLITS_DIR,
    load_yaml_config,
    resolve_device,
    save_checkpoint_metadata,
)


def _count_images(source_entry: str | Path | list[str | Path]) -> int:
    entries = source_entry if isinstance(source_entry, list) else [source_entry]
    total = 0
    for s in entries:
        p = PROJECT_ROOT / s if not Path(s).is_absolute() else Path(s)
        if p.is_dir():
            total += sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        elif p.is_file():
            total += sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    return total


def _rel_str(path: Path | str) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _load_active_dataset_summary() -> dict[str, Any]:
    summary_path = SPLITS_DIR / "dataset_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def inspect_base_checkpoint(model_path: str | Path) -> dict[str, Any]:
    """Inspect initial checkpoint weights SHA-256 and class table without loading onto GPU."""
    p = Path(model_path)
    if not p.is_file():
        return {"exists": False, "path": str(model_path), "sha256": None, "num_classes": 0, "is_coco_base": False, "is_finetuned_pothole": False}

    digest = sha256_file(p)
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    model_obj = ckpt.get("model") if isinstance(ckpt, dict) else None
    names = getattr(model_obj, "names", None) or (ckpt.get("names") if isinstance(ckpt, dict) else {}) or {}
    name_vals = [str(v).lower() for v in (names.values() if isinstance(names, dict) else names)]
    has_pothole = any("pothole" in v for v in name_vals)
    is_coco_base = (len(name_vals) == 80) and not has_pothole
    return {
        "exists": True,
        "path": _rel_str(p),
        "sha256": digest,
        "num_classes": len(name_vals),
        "is_coco_base": is_coco_base,
        "is_finetuned_pothole": has_pothole,
    }


def main(argv: list[str] | None = None) -> None:
    train_cfg = load_yaml_config("training.yaml")
    cfg = train_cfg.get("detector", {})
    aug = cfg.get("augmentation", {})

    default_model = PROJECT_ROOT / cfg.get("base_model", "models/detector/yolo11n.pt")
    default_data = PROJECT_ROOT / cfg.get("data_config", "configs/detector.yaml")
    default_output = PROJECT_ROOT / cfg.get("output_checkpoint", "models/detector/yolo11n_exp_b.pt")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=str(default_model if default_model.exists() else "yolo11n.pt"))
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--epochs", type=int, default=int(cfg.get("epochs", 100)))
    parser.add_argument("--patience", type=int, default=int(cfg.get("patience", 25)))
    parser.add_argument("--batch", type=int, default=int(cfg.get("batch", 16)))
    parser.add_argument("--workers", type=int, default=int(cfg.get("workers", 2)))
    parser.add_argument("--imgsz", type=int, default=int(cfg.get("imgsz", 960)))
    parser.add_argument("--optimizer", type=str, default=str(cfg.get("optimizer", "AdamW")))
    parser.add_argument("--lr0", type=float, default=float(cfg.get("lr0", 0.001)))
    parser.add_argument("--seed", type=int, default=int(cfg.get("seed", 42)))
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=bool(cfg.get("amp", True)))
    parser.add_argument("--device", type=str, default=str(cfg.get("device", "auto")))
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--experiment-profile", type=str, default=None, help="Override experiment profile name for metadata sidecar")
    parser.add_argument("--allow-finetuned-init", action="store_true", help="Permit initializing from an already fine-tuned pothole checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.evaluate_only and args.evaluate_test:
        parser.error("--evaluate-only and --evaluate-test cannot be combined")
    if args.evaluate_only and not Path(args.model).is_file():
        parser.error("--evaluate-only requires --model pointing to an existing checkpoint")

    if str(args.optimizer).strip().lower() == "auto":
        print(
            "[WARNING] optimizer='auto' in Ultralytics ignores custom lr0/momentum settings. "
            "Prefer explicit optimizer='AdamW' with lr0=0.001.",
            file=sys.stderr,
        )

    base_info = inspect_base_checkpoint(args.model)
    if not args.evaluate_only and base_info["exists"]:
        if base_info["is_finetuned_pothole"] and not args.allow_finetuned_init:
            sys.exit(
                f"ERROR: Refusing to initialize training from already fine-tuned pothole checkpoint '{args.model}'. "
                f"Each controlled experiment (A-E) must initialize independently from pretrained base weights "
                f"({cfg.get('base_model', 'models/detector/yolo11n.pt')}). Pass --allow-finetuned-init to override."
            )

    device = resolve_device(args.device)
    resolved_amp = bool(args.amp and device not in {"cpu", "mps"})
    ds_summary = _load_active_dataset_summary()
    exp_profile = args.experiment_profile or ds_summary.get("experiment_profile") or train_cfg.get("active_experiment", "exp_b_roi_90_10")
    exp_spec = (train_cfg.get("experiments") or {}).get(exp_profile, {})
    preproc_mode = ds_summary.get("preprocessing_mode") or exp_spec.get("preprocessing", "roi")
    tile_wide_roi = bool(ds_summary.get("tile_wide_roi") if "tile_wide_roi" in ds_summary else exp_spec.get("tile_wide_roi", False))
    preprocessing_profile = "roi_tiled" if (preproc_mode == "roi" and tile_wide_roi) else preproc_mode
    frozen_info = ds_summary.get("frozen_manifests") or {}

    resolved_hparams = {
        "base_model": base_info["path"] if base_info["exists"] else str(args.model),
        "base_model_sha256": base_info["sha256"],
        "base_model_is_coco_pretrained": base_info["is_coco_base"],
        "experiment_profile": exp_profile,
        "preprocessing_profile": preprocessing_profile,
        "tile_wide_roi": tile_wide_roi,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "amp": resolved_amp,
        "seed": args.seed,
        "workers": args.workers,
        "device": device,
        "augmentation": {
            "mosaic": float(aug.get("mosaic", 0.3)),
            "close_mosaic": int(aug.get("close_mosaic", 15)),
            "mixup": float(aug.get("mixup", 0.0)),
            "cutmix": float(aug.get("cutmix", 0.0)),
            "copy_paste": float(aug.get("copy_paste", 0.0)),
            "degrees": float(aug.get("degrees", 3.0)),
            "translate": float(aug.get("translate", 0.05)),
            "scale": float(aug.get("scale", 0.2)),
            "shear": float(aug.get("shear", 0.0)),
            "perspective": float(aug.get("perspective", 0.0)),
            "fliplr": float(aug.get("fliplr", 0.5)),
            "flipud": float(aug.get("flipud", 0.0)),
            "hsv_h": float(aug.get("hsv_h", 0.01)),
            "hsv_s": float(aug.get("hsv_s", 0.3)),
            "hsv_v": float(aug.get("hsv_v", 0.25)),
        },
    }

    print(f"[Device] Using compute device: {device} (CUDA available: {torch.cuda.is_available()}, AMP: {resolved_amp})")
    print(f"[Base Weights] {resolved_hparams['base_model']} | sha256={resolved_hparams['base_model_sha256']} | coco_base={resolved_hparams['base_model_is_coco_pretrained']}")

    data_yaml_path = args.data.resolve()
    if not data_yaml_path.exists():
        sys.exit(f"ERROR: Dataset YAML config does not exist at {data_yaml_path}")

    print(f"[Dataset] Validating YAML configuration: {data_yaml_path}")
    data_dict = check_det_dataset(str(data_yaml_path))
    n_train = _count_images(data_dict.get("train", []))
    n_val = _count_images(data_dict.get("val", []))
    n_test = _count_images(data_dict.get("test", []))
    print(f"  Classes: {data_dict.get('names')}")
    print(f"  Total train images: {n_train}")
    print(f"  Total val images:   {n_val}")
    print(f"  Total test images:  {n_test} (held-out, protected)")
    print("[Resolved Hyperparameters]")
    print(json.dumps(resolved_hparams, indent=2))

    if args.dry_run:
        print("\n[Dry-run passed successfully. Dataset YAML parsed; inspect resolved counts & hyperparameters before training.]")
        return

    if args.evaluate_only:
        if args.split == "test" and n_test == 0:
            sys.exit("ERROR: Held-out test split is unavailable or empty.")
        if args.split == "val" and n_val == 0:
            sys.exit("ERROR: Validation split is empty.")
        print(f"\n[Evaluation only] Loading {args.model} for {args.split} evaluation; no training.")
        metrics = YOLO(args.model).val(data=str(data_yaml_path), split=args.split, device=device, imgsz=args.imgsz)
        if hasattr(metrics, "box"):
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall:    {metrics.box.mr:.4f}")
            print(f"  mAP50:     {metrics.box.map50:.4f}")
            print(f"  mAP50-95:  {metrics.box.map:.4f}")
        return

    run_name = args.experiment_name or datetime.now(timezone.utc).strftime("pothole_%Y%m%dT%H%M%S_%fZ")
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        parser.error("--experiment-name must be a single directory name")
    runs_root = RUNS_DIR / "detector"
    run_dir_target = runs_root / run_name
    if run_dir_target.exists() and not args.overwrite:
        sys.exit(f"ERROR: Run already exists: {run_dir_target}. Choose another --experiment-name.")
    if args.output.exists() and not args.overwrite:
        sys.exit(f"ERROR: Checkpoint already exists: {args.output}. Choose a versioned --output, or pass --overwrite.")
    if args.evaluate_test and n_test == 0:
        sys.exit("ERROR: --evaluate-test requested but held-out test split is missing or empty.")

    aug_resolved = resolved_hparams["augmentation"]
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        amp=resolved_amp,
        mosaic=aug_resolved["mosaic"],
        mixup=aug_resolved["mixup"],
        cutmix=aug_resolved["cutmix"],
        copy_paste=aug_resolved["copy_paste"],
        degrees=aug_resolved["degrees"],
        translate=aug_resolved["translate"],
        scale=aug_resolved["scale"],
        shear=aug_resolved["shear"],
        perspective=aug_resolved["perspective"],
        fliplr=aug_resolved["fliplr"],
        flipud=aug_resolved["flipud"],
        hsv_h=aug_resolved["hsv_h"],
        hsv_s=aug_resolved["hsv_s"],
        hsv_v=aug_resolved["hsv_v"],
        close_mosaic=aug_resolved["close_mosaic"],
        project=str(runs_root),
        name=run_name,
        exist_ok=args.overwrite,
    )

    val_summary: dict[str, float] = {}
    val_metrics = model.val(data=str(data_yaml_path), split="val", device=device)
    if hasattr(val_metrics, "box"):
        val_summary = {
            "precision": round(float(val_metrics.box.mp), 4),
            "recall": round(float(val_metrics.box.mr), 4),
            "mAP50": round(float(val_metrics.box.map50), 4),
            "mAP50_95": round(float(val_metrics.box.map), 4),
        }
        print(
            f"\n--- Validation Metrics ---\n"
            f"  P={val_summary['precision']:.4f} R={val_summary['recall']:.4f} "
            f"mAP50={val_summary['mAP50']:.4f} mAP50-95={val_summary['mAP50_95']:.4f}"
        )

    if args.evaluate_test and n_test > 0:
        test_metrics = model.val(data=str(data_yaml_path), split="test", device=device)
        if hasattr(test_metrics, "box"):
            print(
                f"\n--- Test Metrics ---\n"
                f"  P={test_metrics.box.mp:.4f} R={test_metrics.box.mr:.4f} "
                f"mAP50={test_metrics.box.map50:.4f} mAP50-95={test_metrics.box.map:.4f}"
            )

    best_weights_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best_weights_path.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with best_weights_path.open("rb") as src, args.output.open("wb" if args.overwrite else "xb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"\n[OK] Copied best model weights to: {args.output}")

        meta_payload = {
            "checkpoint": _rel_str(args.output),
            "architecture": Path(str(args.model)).stem,
            "base_model": resolved_hparams["base_model"],
            "base_model_sha256": resolved_hparams["base_model_sha256"],
            "experiment_name": run_name,
            "experiment_profile": exp_profile,
            "dataset_version": ds_summary.get("dataset_version") or (train_cfg.get("dataset") or {}).get("version", "curated_v1"),
            "data_config": _rel_str(data_yaml_path),
            "manifests": {
                "master_manifest": "data/manifests/master_manifest.csv",
                "dataset_summary": "data/splits/dataset_summary.json",
                "frozen_manifest": frozen_info.get("manifest_csv"),
                "frozen_manifest_sha256": frozen_info.get("manifest_sha256"),
                "frozen_summary": frozen_info.get("summary_json"),
            },
            "split_counts": {"train": n_train, "val": n_val, "test": n_test},
            "preprocessing_profile": preprocessing_profile,
            "tile_wide_roi": tile_wide_roi,
            "source_rois": train_cfg.get("source_rois", {}),
            "input_imgsz": args.imgsz,
            "hyperparameters": resolved_hparams,
            "val_metrics": val_summary,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = save_checkpoint_metadata(args.output, meta_payload)
        print(f"[OK] Saved checkpoint preprocessing metadata -> {meta_path}")


if __name__ == "__main__":
    main()
