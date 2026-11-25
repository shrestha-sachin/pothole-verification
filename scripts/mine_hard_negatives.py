"""Mine hard-negative (false positive) crops strictly from the training split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.inference import PotholeDetector
from src.utils.config import PROCESSED_DIR, load_yaml_config, resolve_device
from src.verifier.preprocessing import expand_and_clamp_box


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(1e-6, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


def main(argv: list[str] | None = None) -> None:
    inf_cfg = load_yaml_config("inference.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / inf_cfg.get("detector_checkpoint", "models/detector/yolo26x_us_v5_960.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=int(inf_cfg.get("imgsz", 960)))
    parser.add_argument("--max-crops", type=int, default=500)
    args = parser.parse_args(argv)

    # Strictly use TRAIN split only; never mine val or test!
    train_imgs = sorted((PROCESSED_DIR / "detector" / "images" / "train").glob("*.jpg"))
    train_lbls = PROCESSED_DIR / "detector" / "labels" / "train"
    out_dir = PROCESSED_DIR / "verifier" / "train" / "non_pothole"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PotholeDetector(args.model, conf=args.conf, imgsz=args.imgsz, device=resolve_device("auto"))
    mined = 0

    for img_p in train_imgs:
        if mined >= args.max_crops:
            break
        with Image.open(img_p) as im:
            rgb = im.convert("RGB")
            iw, ih = rgb.size
        lbl_p = train_lbls / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for ln in lbl_p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                _, cx, cy, bw, bh = map(float, ln.split())
                gt_boxes.append(((cx - bw / 2) * iw, (cy - bh / 2) * ih, (cx + bw / 2) * iw, (cy + bh / 2) * ih))

        frame_bgr = cv2.imread(str(img_p))
        if frame_bgr is None:
            continue
        preds = detector.run_on_frame(frame_bgr)
        for idx, p in enumerate(preds):
            if mined >= args.max_crops:
                break
            pb = (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"]))
            if any(_iou(pb, gb) >= 0.15 for gb in gt_boxes):
                continue
            ex1, ey1, ex2, ey2 = expand_and_clamp_box(*pb, iw, ih, pad_frac=0.25)
            if ex2 > ex1 and ey2 > ey1:
                rgb.crop((ex1, ey1, ex2, ey2)).save(out_dir / f"hn_{img_p.stem}_{idx:02d}.jpg", quality=92)
                mined += 1

    print(f"[OK] Mined {mined} hard-negative crops from train split into {out_dir}")


if __name__ == "__main__":
    main()
