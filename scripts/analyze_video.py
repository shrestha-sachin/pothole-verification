"""Run end-to-end video pothole detection, BoT-SORT tracking, and visual verification."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.detector.video_sampler import iter_frames
from src.evaluation.policies import PolicyConfig, apply_policies
from src.tracking.event_grouper import EventGrouper
from src.utils.config import EVALUATIONS_DIR, PREDICTIONS_DIR, load_yaml_config, resolve_device
from src.verifier.interface import load_verifier


def main(argv: list[str] | None = None) -> None:
    inf_cfg = load_yaml_config("inference.yaml")
    trk_cfg = load_yaml_config("tracking.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / inf_cfg.get("detector_checkpoint", "models/detector/yolo26x_us_v5_960.pt"))
    parser.add_argument("--verifier", type=Path, default=PROJECT_ROOT / inf_cfg.get("verifier_checkpoint", "models/verifier/verifier_v1.pt"))
    parser.add_argument("--conf", type=float, default=float(inf_cfg.get("confidence_threshold", 0.25)))
    parser.add_argument("--iou", type=float, default=float(inf_cfg.get("iou_threshold", 0.50)))
    parser.add_argument("--imgsz", type=int, default=int(inf_cfg.get("imgsz", 960)))
    parser.add_argument("--sample-fps", type=float, default=float(inf_cfg.get("sample_fps", 2.0)))
    parser.add_argument("--device", type=str, default=str(inf_cfg.get("device", "auto")))
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.video.exists():
        sys.exit(f"ERROR: Video not found: {args.video}")

    device = resolve_device(args.device)
    model = YOLO(str(args.model))
    verifier = load_verifier(args.verifier) if args.verifier.exists() else None

    video_id = args.video.stem
    grouper = EventGrouper(
        video_id=video_id,
        policy_b_min_obs=int(trk_cfg.get("min_track_frames", 2)),
    )
    frame_iter = iter_frames(args.video, target_fps=args.sample_fps)
    events = grouper.process_video_stream(
        frame_iter=frame_iter,
        model=model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        evidence_root=EVALUATIONS_DIR,
        video_path=args.video,
    )

    policy_cfg = PolicyConfig(
        min_obs=int(trk_cfg.get("min_track_frames", 2)),
        verifier_threshold=float(inf_cfg.get("verifier_threshold", 0.50)),
    )
    for ev in events:
        if verifier is not None and ev.crop_path and Path(ev.crop_path).exists():
            crop_bgr = cv2.imread(str(ev.crop_path))
            if crop_bgr is not None:
                vr = verifier.score_crop(crop_bgr)
                ev.policy_c_score = float(getattr(vr, "pothole_prob", getattr(vr, "score", 0.0)))
    events = apply_policies(events, policy_cfg)
    results = [dataclasses.asdict(ev) for ev in events]

    out = args.output_json or (PREDICTIONS_DIR / f"{video_id}_events.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[OK] Detected {len(results)} tracked pothole events -> {out}")


if __name__ == "__main__":
    main()
