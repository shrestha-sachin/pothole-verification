"""Tests for canonical repository structure, ROI transforms, quality checks, and dataset validation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate import evaluate_predictions_original_space, run_original_space_evaluation
from scripts.train_detector import inspect_base_checkpoint, main as train_detector_main
from src.data.validation import validate_detector_processed, validate_yolo_label_file
from src.detector.inference import predict_frame_detections
from src.preprocessing.quality import QualityThresholds, assess_image_quality, compute_effective_yolo_box_px
from src.preprocessing.roi import SourceROI, apply_roi_crop, generate_horizontal_tiles, map_predictions_to_original
from src.tracking.event_grouper import EventGrouper
from src.utils.config import CONFIGS_DIR, DETECTOR_MODELS_DIR, MANIFESTS_DIR, PROCESSED_DIR, load_checkpoint_metadata, load_yaml_config, save_checkpoint_metadata


def test_configs_are_portable_without_hardcoded_absolute_paths():
    for name in ("detector.yaml", "verifier.yaml", "training.yaml", "inference.yaml", "tracking.yaml"):
        cfg_path = CONFIGS_DIR / name
        assert cfg_path.exists(), f"Missing {cfg_path}"
        txt = cfg_path.read_text(encoding="utf-8")
        assert "C:/" not in txt and "C:\\" not in txt, f"Hardcoded Windows path in {name}"
        assert "/Users/" not in txt, f"Hardcoded macOS path in {name}"
        data = load_yaml_config(name)
        assert isinstance(data, dict)


def test_detector_yaml_single_pothole_class():
    cfg = load_yaml_config("detector.yaml")
    assert cfg["names"] == {0: "pothole"}


def test_validate_yolo_label_file_accepts_empty_and_rejects_bad_class(tmp_path: Path):
    empty_lbl = tmp_path / "empty.txt"
    empty_lbl.write_text("", encoding="utf-8")
    boxes, errs = validate_yolo_label_file(empty_lbl)
    assert boxes == [] and errs == []

    bad_cls = tmp_path / "bad.txt"
    bad_cls.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    _, errs = validate_yolo_label_file(bad_cls)
    assert any("invalid_class_1" in e for e in errs)


def test_roi_transform_and_back_mapping_roundtrip():
    img = Image.new("RGB", (1000, 1000), color=(120, 120, 120))
    roi = SourceROI(x1=0.1, y1=0.5, x2=0.9, y2=1.0)
    # Box centered at (500, 750) with size (100, 100) in 1000x1000 -> norm (0.5, 0.75, 0.1, 0.1)
    res = apply_roi_crop(img, [(0, 0.5, 0.75, 0.1, 0.1)], roi)
    assert res.crop_wh == (800, 500)
    assert len(res.boxes) == 1
    _, ncx, ncy, nbw, nbh = res.boxes[0]
    assert abs(ncx - 0.5) < 1e-4
    assert abs(ncy - 0.5) < 1e-4
    assert abs(nbw - 0.125) < 1e-4
    assert abs(nbh - 0.2) < 1e-4

    # Map prediction in ROI coords back to original 1000x1000 coords
    mapped = map_predictions_to_original([{"x1": 350, "y1": 200, "x2": 450, "y2": 300, "conf": 0.9}], res.crop_box_px)
    assert mapped[0]["x1"] == 450.0 and mapped[0]["y1"] == 700.0
    assert mapped[0]["x2"] == 550.0 and mapped[0]["y2"] == 800.0


def test_horizontal_tiling_nms_deduplicates_overlap():
    img = Image.new("RGB", (3200, 800), color=(100, 100, 100))
    tiles = generate_horizontal_tiles(img, [(0, 0.5, 0.5, 0.05, 0.2)], (0, 1000, 3200, 1800))
    assert len(tiles) == 2
    # Both tiles contain the center box; verify mapping back & NMS merges duplicate predictions
    all_preds = []
    for t in tiles:
        _, cx, cy, bw, bh = t.boxes[0]
        tw, th = t.crop_wh
        all_preds.extend(map_predictions_to_original([{
            "x1": (cx - bw / 2) * tw,
            "y1": (cy - bh / 2) * th,
            "x2": (cx + bw / 2) * tw,
            "y2": (cy + bh / 2) * th,
            "conf": 0.85,
        }], t.crop_box_px))
    merged = map_predictions_to_original(all_preds, (0, 0, 0, 0), nms_iou_thresh=0.5)
    assert len(merged) == 1


def test_quality_assessment_rejects_corrupt_and_computes_effective_size(tmp_path: Path):
    bad_p = tmp_path / "missing.jpg"
    q_bad = assess_image_quality(bad_p)
    assert not q_bad.valid_image and "corrupt_image" in q_bad.flags

    ew, eh, scale = compute_effective_yolo_box_px(3840, 2160, 0.025, 0.02, 960)
    assert abs(scale - 0.25) < 1e-6
    assert abs(ew - 24.0) < 1e-4


def test_canonical_processed_detector_integrity():
    det_dir = PROCESSED_DIR / "detector"
    if not det_dir.exists():
        return
    report = validate_detector_processed(det_dir)
    assert report["valid"], f"Validation failed: {report}"
    assert report["splits"]["train"]["images"] > 0
    assert report["splits"]["val"]["images"] > 0
    assert report["splits"]["test"]["images"] > 0
    assert (det_dir / "dataset.yaml").exists()


def test_checkpoint_metadata_save_load_roundtrip(tmp_path: Path):
    ckpt = tmp_path / "dummy_exp_b.pt"
    ckpt.write_bytes(b"weights")
    # Default when sidecar missing
    assert load_checkpoint_metadata(ckpt)["preprocessing_profile"] == "full_frame"

    meta_p = save_checkpoint_metadata(
        ckpt,
        {
            "preprocessing_profile": "roi",
            "tile_wide_roi": False,
            "input_imgsz": 960,
            "source_rois": {"local_dashcam": [0.04, 0.64, 0.88, 0.992]},
        },
    )
    assert meta_p.exists()
    loaded = load_checkpoint_metadata(ckpt)
    assert loaded["preprocessing_profile"] == "roi"
    assert loaded["input_imgsz"] == 960
    assert "package_versions" in loaded


def test_train_detector_dry_run_uses_explicit_adamw_and_lr0(capsys):
    det_dir = PROCESSED_DIR / "detector"
    if not det_dir.exists():
        return
    train_detector_main(["--dry-run"])
    out = capsys.readouterr().out
    assert '"optimizer": "AdamW"' in out
    assert '"lr0": 0.001' in out
    assert '"epochs": 100' in out
    assert '"batch": 16' in out
    assert '"imgsz": 960' in out


def test_original_space_evaluation_counts_roi_excluded_gt_as_miss():
    gt = {
        "img_1": [(100.0, 700.0, 200.0, 800.0), (50.0, 50.0, 100.0, 100.0)],
    }
    # Only the in-ROI box is predicted; the top sky-border GT box is excluded -> FN=1
    preds = {
        "img_1": [{"x1": 100.0, "y1": 700.0, "x2": 200.0, "y2": 800.0, "conf": 0.90}],
    }
    res = evaluate_predictions_original_space(preds, gt, conf_threshold=0.25, iou_threshold=0.50, roi_excluded_gt_count=1)
    assert res["tp"] == 1
    assert res["fn"] == 1
    assert res["fp"] == 0
    assert res["roi_excluded_gt_boxes"] == 1
    assert res["recall"] == 0.5
    assert res["precision"] == 1.0


def test_assisted_proposals_exclude_protected_test_split():
    proposals_path = MANIFESTS_DIR / "assisted_proposals.json"
    master_path = MANIFESTS_DIR / "master_manifest.csv"
    if not proposals_path.exists() or not master_path.exists():
        return
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    with master_path.open(encoding="utf-8", newline="") as fh:
        test_ids = {r["source_image_id"] for r in csv.DictReader(fh) if r["split"] == "test"}
    overlap = set(proposals.keys()) & test_ids
    assert overlap == set(), f"Protected test images leaked into assisted_proposals: {overlap}"
    assert len(proposals) >= 700


def test_verifier_manifest_split_integrity_and_hard_negatives():
    v_manifest = PROCESSED_DIR / "verifier" / "verifier_manifest.csv"
    if not v_manifest.exists():
        return
    with v_manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    # Ensure no source_image_id or group_id crosses splits
    img_splits: dict[str, set[str]] = {}
    grp_splits: dict[str, set[str]] = {}
    subtypes: set[str] = set()
    for r in rows:
        sp = r["split"]
        img_splits.setdefault(r["source_image_id"], set()).add(sp)
        if r.get("group_id"):
            grp_splits.setdefault(r["group_id"], set()).add(sp)
        if r["label"] == "non_pothole":
            subtypes.add(r.get("subtype", ""))
        assert float(r.get("pad_frac", 0.0)) == 0.15
    assert all(len(s) == 1 for s in img_splits.values())
    assert all(len(s) == 1 for s in grp_splits.values())
    assert len(subtypes - {"", "other"}) >= 2, f"Expected diverse hard negative subtypes, got {subtypes}"


class _MockTensorVal:
    def __init__(self, val):
        self._val = val

    def item(self):
        return float(self._val)

    def tolist(self):
        return list(self._val)


class _MockBox:
    def __init__(self, x1: float, y1: float, x2: float, y2: float, conf: float, cls_id: int = 0):
        self.xyxy = [_MockTensorVal([x1, y1, x2, y2])]
        self.conf = [_MockTensorVal(conf)]
        self.cls = [_MockTensorVal(cls_id)]


def test_tiled_video_inference_nms_merges_before_botsort_tracking(tmp_path: Path):
    import cv2
    from src.video.annotated_renderer import AnnotatedVideoRenderer

    roi = SourceROI(x1=0.04, y1=0.64, x2=0.88, y2=0.992)
    # 3840x2160 ROI is [154, 1382, 3379, 2143] -> two tiles of width 1854 ending at x=2008 and starting at x=1525.
    # Global box [1700, 1680, 1800, 1720] lies inside BOTH tiles' overlap zone!
    seen_shapes: list[tuple[int, int]] = []

    class _MockModel:
        names = {0: "pothole"}
        predictor = None

        def predict(self, sub_bgr, **kwargs):
            h, w = sub_bgr.shape[:2]
            seen_shapes.append((w, h))
            # Tile 1 starts at x=154 -> local x1=1700-154=1546
            # Tile 2 starts at x=1525 -> local x1=1700-1525=175
            if len(seen_shapes) % 2 == 1:
                box = _MockBox(1546.0, 298.0, 1646.0, 338.0, conf=0.88)
            else:
                box = _MockBox(175.0, 298.0, 275.0, 338.0, conf=0.81)
            return [type("_Res", (), {"boxes": [box]})()]

    model = _MockModel()
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    preds = predict_frame_detections(model, frame, roi=roi, tile_wide_roi=True, merge_iou=0.50, conf=0.25)
    assert seen_shapes == [(1854, 761), (1854, 761)]
    assert len(preds) == 1
    assert preds[0]["x1"] == 1700.0 and preds[0]["y1"] == 1680.0
    assert abs(preds[0]["conf"] - 0.88) < 1e-6

    grouper = EventGrouper(video_id="tiled_vid_test", policy_b_min_obs=2, policy_b_window_frames=5)
    stream = [(i, i * 100, frame.copy()) for i in range(3)]
    events = grouper.process_video_stream(
        stream, model=model, conf=0.25, iou=0.5, imgsz=960, device="cpu", roi=roi, tile_wide_roi=True, merge_iou=0.50
    )
    assert len(events) == 1, f"Duplicate overlap tracks not merged before BoT-SORT: {events}"
    assert events[0].observation_count == 3
    assert events[0].rep_x1 == 1700 and events[0].rep_y1 == 1680

    # Verify AnnotatedVideoRenderer also applies identical tiled ROI + original-frame NMS + BoT-SORT tracking
    vid_p = tmp_path / "wide_tiled.mp4"
    wr = cv2.VideoWriter(str(vid_p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (3840, 2160))
    for _ in range(2):
        wr.write(frame)
    wr.release()

    renderer = AnnotatedVideoRenderer(output_dir=tmp_path / "rendered")
    meta = renderer.render_annotated_video(
        vid_p, model=model, conf_threshold=0.25, iou_threshold=0.50, imgsz=960,
        device="cpu", use_tracking=True, roi=roi, preprocessing_profile="roi_tiled", tile_wide_roi=True,
    )
    assert meta.tile_wide_roi is True
    assert meta.total_detections_drawn == 2
    assert len(meta.unique_track_ids) == 1


def test_original_space_map_separates_low_conf_floor_from_fixed_threshold(tmp_path: Path):
    gt = {
        "img_1": [(10.0, 10.0, 50.0, 50.0)],
        "img_2": [(20.0, 20.0, 60.0, 60.0)],
    }
    preds = {
        "img_1": [{"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0, "conf": 0.85}],
        "img_2": [
            {"x1": 200.0, "y1": 200.0, "x2": 240.0, "y2": 240.0, "conf": 0.90},  # FP at 0.90
            {"x1": 20.0, "y1": 20.0, "x2": 60.0, "y2": 60.0, "conf": 0.10},      # TP below 0.25 operating threshold
        ],
    }
    res = evaluate_predictions_original_space(preds, gt, conf_threshold=0.25, conf_floor=0.001, iou_threshold=0.50)
    # Fixed operating point @ conf >= 0.25 only counts 1 TP, 1 FP, 1 FN -> Recall=0.5, Precision=0.5
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 1
    assert res["recall"] == 0.5 and res["precision"] == 0.5
    assert res["false_positives_per_image"] == 0.5
    # Standard mAP integrates down to conf_floor=0.001, recovering the conf=0.10 TP
    assert res["mAP50"] > 0.65
    assert res["mAP50_95"] > 0.65

    with pytest.raises(ValueError, match="conf_floor"):
        run_original_space_evaluation(DETECTOR_MODELS_DIR / "yolo11n.pt", "val", conf_floor=0.25)


def test_base_checkpoint_verification_blocks_finetuned_init():
    base_pt = DETECTOR_MODELS_DIR / "yolo11n.pt"
    if base_pt.exists():
        info = inspect_base_checkpoint(base_pt)
        assert info["is_coco_base"] is True
        assert info["is_finetuned_pothole"] is False
        assert info["sha256"]

    ft_pt = DETECTOR_MODELS_DIR / "yolo26x_us_v5_960.pt"
    if ft_pt.exists():
        info_ft = inspect_base_checkpoint(ft_pt)
        assert info_ft["is_finetuned_pothole"] is True
        with pytest.raises(SystemExit):
            train_detector_main(["--model", str(ft_pt), "--dry-run"])


def test_frozen_versioned_manifests_and_physical_distinction():
    train_cfg = load_yaml_config("training.yaml")
    ver = str((train_cfg.get("dataset") or {}).get("version", "curated_v1"))
    for exp in ("exp_a_full_90_10", "exp_b_roi_90_10", "exp_c_tiled_90_10"):
        m_p = MANIFESTS_DIR / f"{ver}_{exp}_manifest.csv"
        s_p = MANIFESTS_DIR / f"{ver}_{exp}_summary.json"
        if not m_p.exists():
            continue
        assert s_p.exists(), f"Missing frozen summary {s_p}"
        s_doc = json.loads(s_p.read_text(encoding="utf-8"))
        assert s_doc["experiment_profile"] == exp
        obj_phys = s_doc.get("object_vs_physical_counts", {})
        assert "local_dashcam" in obj_phys
        assert obj_phys["local_dashcam"]["bounding_box_instances"] > obj_phys["local_dashcam"]["estimated_temporal_encounters_unverified"]
        assert obj_phys["local_dashcam"]["verified_unique_physical_potholes"] == 0
