"""Canonical dataset preparation CLI supporting controlled experiments A-E."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pipeline import build_curated_dataset
from src.data.report import generate_dataset_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=[
            "exp_a_full_90_10",
            "exp_b_roi_90_10",
            "exp_c_tiled_90_10",
            "exp_d_best_1280",
            "exp_e_best_80_20",
        ],
        help="Experiment configuration profile from configs/training.yaml (default: active_experiment)",
    )
    parser.add_argument("--prepare-verifier", action="store_true", help="Also build data/processed/verifier crops")
    parser.add_argument("--report", action="store_true", help="Generate markdown quality report and visual previews")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument(
        "--force-refreeze",
        action="store_true",
        help="Explicitly overwrite an already-frozen versioned manifest snapshot in data/manifests/",
    )
    args = parser.parse_args(argv)

    summary = build_curated_dataset(
        experiment=args.experiment,
        prepare_verifier=args.prepare_verifier,
        seed=args.seed,
        force_refreeze=args.force_refreeze,
    )
    print(json.dumps(summary, indent=2))
    if args.report:
        report_p = generate_dataset_report()
        print(f"\n[OK] Dataset report & visual previews written to: {report_p}")
    if not summary.get("validation_passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
