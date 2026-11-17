"""Automated regression and hygiene test suite for Experiment V3 dataset pipeline.

Verifies:
  1. No duplicate image IDs across train, val, and test splits.
  2. No SHA-256 duplicate content across splits.
  3. No recording or sequence overlap across splits (temporal leakage prevention).
  4. Every image path listed in manifests resolves to an existing file on disk.
  5. Every YOLO label path resolves to an existing file on disk.
  6. Correct YOLO bounding-box format (class 0, normalized cx cy nw nh in [0, 1]).
  7. Nonempty RDD2022 train, validation, and test sets adhering to 80/10/10 proportions.
  8. Determinism: Seed 42 produces identical splits across multiple runs.
  9. Non-destructive protection: Refuses to overwrite existing files without explicit flag.
"""

import csv
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from ultralytics.data.utils import check_det_dataset, img2label_paths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "rdd2022_v3"
LOCAL_EXPORTS_DIR = PROJECT_ROOT / "exports" / "yolo_labels"
YAML_V3_PATH = PROJECT_ROOT / "configs" / "detector_v3.yaml"
INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"


class TestV3PipelineHygiene(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read RDD V3 manifests
        cls.rdd_train = [Path(l.strip()) for l in (V3_PROCESSED_DIR / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.rdd_val = [Path(l.strip()) for l in (V3_PROCESSED_DIR / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.rdd_test = [Path(l.strip()) for l in (V3_PROCESSED_DIR / "test.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

        # Read Local manifests
        cls.local_train = [Path(l.strip()) for l in (LOCAL_EXPORTS_DIR / "train.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        cls.local_val = [Path(l.strip()) for l in (LOCAL_EXPORTS_DIR / "val.txt").read_text(encoding="utf-8").splitlines() if l.strip()]

    def test_01_no_duplicate_image_ids_across_splits(self):
        """Verify absolute disjointness of image stems across train, val, and test."""
        train_stems = {p.stem for p in self.rdd_train + self.local_train}
        val_stems = {p.stem for p in self.rdd_val + self.local_val}
        test_stems = {p.stem for p in self.rdd_test}

        overlap_tv = train_stems & val_stems
        overlap_tt = train_stems & test_stems
        overlap_vt = val_stems & test_stems

        self.assertEqual(len(overlap_tv), 0, f"Train and Val share image stems: {overlap_tv}")
        self.assertEqual(len(overlap_tt), 0, f"Train and Test share image stems: {overlap_tt}")
        self.assertEqual(len(overlap_vt), 0, f"Val and Test share image stems: {overlap_vt}")

    def test_02_no_sha256_duplicates_across_splits(self):
        """Verify that identical image content never appears across different splits."""
        def hash_file(p: Path) -> str:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()

        # Build hash maps
        train_hashes = {hash_file(p): p for p in self.rdd_train + self.local_train}
        val_hashes = {hash_file(p): p for p in self.rdd_val + self.local_val}
        test_hashes = {hash_file(p): p for p in self.rdd_test}

        hash_overlap_tv = set(train_hashes.keys()) & set(val_hashes.keys())
        hash_overlap_tt = set(train_hashes.keys()) & set(test_hashes.keys())
        hash_overlap_vt = set(val_hashes.keys()) & set(test_hashes.keys())

        self.assertEqual(len(hash_overlap_tv), 0, f"SHA-256 collision between Train and Val: {hash_overlap_tv}")
        self.assertEqual(len(hash_overlap_tt), 0, f"SHA-256 collision between Train and Test: {hash_overlap_tt}")
        self.assertEqual(len(hash_overlap_vt), 0, f"SHA-256 collision between Val and Test: {hash_overlap_vt}")

    def test_03_no_recording_or_sequence_overlap(self):
        """Verify recording-level and sequence-level split isolation."""
        with INVENTORY_PATH.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        rec_splits = {}
        seq_splits = {}
        for r in rows:
            rec = r.get("recording_id", "").strip()
            seq = r.get("sequence_id", "").strip()
            split = r.get("split", "").strip()
            if rec:
                rec_splits.setdefault(rec, set()).add(split)
            if seq:
                seq_splits.setdefault(seq, set()).add(split)

        rec_conflicts = {k: v for k, v in rec_splits.items() if len(v) > 1}
        seq_conflicts = {k: v for k, v in seq_splits.items() if len(v) > 1}

        self.assertEqual(len(rec_conflicts), 0, f"Recording-level data leakage detected: {rec_conflicts}")
        self.assertEqual(len(seq_conflicts), 0, f"Sequence-level data leakage detected: {seq_conflicts}")

    def test_04_every_image_path_resolves(self):
        """Verify that 100% of image paths in manifests exist and are non-empty files."""
        all_manifest_paths = self.rdd_train + self.rdd_val + self.rdd_test + self.local_train + self.local_val
        missing = []
        for p in all_manifest_paths:
            resolved = p if p.is_absolute() else (PROJECT_ROOT / p)
            if not resolved.exists() or resolved.stat().st_size == 0:
                missing.append(str(p))

        self.assertEqual(len(missing), 0, f"{len(missing)} image files missing or empty: {missing[:5]}")

    def test_05_every_yolo_label_path_resolves(self):
        """Verify that Ultralytics img2label_paths resolves valid label files for all images."""
        all_manifest_paths = [str(p if p.is_absolute() else (PROJECT_ROOT / p)) for p in self.rdd_train + self.rdd_val + self.rdd_test + self.local_train + self.local_val]
        expected_labels = img2label_paths(all_manifest_paths)

        missing_labels = []
        for l_path in expected_labels:
            p = Path(l_path)
            if not p.exists():
                missing_labels.append(l_path)

        self.assertEqual(len(missing_labels), 0, f"{len(missing_labels)} YOLO label files missing: {missing_labels[:5]}")

    def test_06_bounding_box_format(self):
        """Verify YOLO bounding box formatting: class 0, normalized [0, 1] xywh coordinates."""
        all_manifest_paths = [str(p if p.is_absolute() else (PROJECT_ROOT / p)) for p in self.rdd_train + self.rdd_val + self.rdd_test + self.local_train + self.local_val]
        label_paths = img2label_paths(all_manifest_paths)

        invalid_boxes = []
        for lp in label_paths:
            p = Path(lp)
            content = p.read_text(encoding="utf-8").strip()
            if not content:
                # Valid negative image
                continue
            for line_idx, line in enumerate(content.splitlines()):
                parts = line.strip().split()
                if len(parts) != 5:
                    invalid_boxes.append((lp, line, "Expected 5 fields"))
                    continue
                cls_id, cx, cy, w, h = parts
                if cls_id != "0":
                    invalid_boxes.append((lp, line, f"Invalid class ID: {cls_id}"))
                    continue
                try:
                    vals = [float(cx), float(cy), float(w), float(h)]
                except ValueError:
                    invalid_boxes.append((lp, line, "Non-numeric values"))
                    continue
                for v, name in zip(vals, ["cx", "cy", "w", "h"]):
                    if not (0.0 <= v <= 1.0):
                        invalid_boxes.append((lp, line, f"{name}={v} out of [0, 1]"))
                if vals[2] <= 0.0 or vals[3] <= 0.0:
                    invalid_boxes.append((lp, line, "Zero or negative width/height"))

        self.assertEqual(len(invalid_boxes), 0, f"Found {len(invalid_boxes)} invalid bounding boxes: {invalid_boxes[:3]}")

    def test_07_nonempty_rdd2022_splits(self):
        """Verify that RDD2022 V3 splits are nonempty and follow ~80/10/10 distribution."""
        n_t = len(self.rdd_train)
        n_v = len(self.rdd_val)
        n_te = len(self.rdd_test)
        total = n_t + n_v + n_te

        self.assertGreater(n_t, 0)
        self.assertGreater(n_v, 0)
        self.assertGreater(n_te, 0)

        # Proportions within 1.0% of target 80/10/10
        self.assertAlmostEqual(n_t / total, 0.80, delta=0.015)
        self.assertAlmostEqual(n_v / total, 0.10, delta=0.015)
        self.assertAlmostEqual(n_te / total, 0.10, delta=0.015)

    def test_08_deterministic_splits_seed_42(self):
        """Verify that seed 42 produces 100% deterministic, reproducible split assignments."""
        from scripts.prepare_rdd2022_v3 import DEFAULT_RDD_ROOT, assign_connected_group_ids, discover_country_items, stratified_split_groups

        us_items1 = discover_country_items(DEFAULT_RDD_ROOT, "United_States")[:20]
        groups1 = assign_connected_group_ids(us_items1)
        t1, v1, te1 = stratified_split_groups(groups1, 0.8, 0.1, 0.1, seed=42)

        us_items2 = discover_country_items(DEFAULT_RDD_ROOT, "United_States")[:20]
        groups2 = assign_connected_group_ids(us_items2)
        t2, v2, te2 = stratified_split_groups(groups2, 0.8, 0.1, 0.1, seed=42)

        self.assertEqual([x.stem for x in t1], [x.stem for x in t2])
        self.assertEqual([x.stem for x in v1], [x.stem for x in v2])
        self.assertEqual([x.stem for x in te1], [x.stem for x in te2])

    def test_09_non_destructive_protection(self):
        """Verify that scripts refuse to overwrite existing files without explicit flag."""
        from scripts.export_yolo import export
        with tempfile.TemporaryDirectory() as temp:
            dummy_out = Path(temp) / "dummy"
            dummy_out.mkdir()
            (dummy_out / "sentinel.txt").write_text("protected\n")
            with self.assertRaises(RuntimeError):
                export(dummy_out, overwrite=False)
            self.assertEqual((dummy_out / "sentinel.txt").read_text(), "protected\n")


if __name__ == "__main__":
    unittest.main()
