"""Validate the canonical processed detector dataset and YOLO config."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.data.utils import check_det_dataset
from src.data.validation import validate_detector_processed
from src.utils.config import CONFIGS_DIR, PROCESSED_DIR


def main() -> None:
    det_dir = PROCESSED_DIR / "detector"
    report = validate_detector_processed(det_dir)
    cfg_path = CONFIGS_DIR / "detector.yaml"
    ds_info = check_det_dataset(str(cfg_path))
    report["ultralytics_names"] = ds_info.get("names")
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
