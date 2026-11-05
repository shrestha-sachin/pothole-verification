"""Unit tests for verifier training hygiene, split isolation, and corrupt sample rejection.

Verifies:
  - Strict split isolation: train only uses 'train', val only uses 'val' / 'validation'.
  - Pilot, test, blank, and unknown splits are excluded from training.
  - Refuses to train on full-frame inventory files.
  - Corrupt or missing crop images raise FileNotFoundError/CorruptSampleError (no silent zeros).
  - Unknown labels are rejected with ValueError.
  - check_split_independence detects sequence / video leakage across splits.
  - evaluate_verifier computes loss, accuracy, precision, recall, F1, and confusion matrix.
"""

import csv
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.verifier.train import (
    CorruptSampleError,
    PotholeCropDataset,
    check_split_independence,
    evaluate_verifier,
    load_dataset_splits,
)


class DummyModel(nn.Module):
    def __init__(self, constant_pred_class: int = 1):
        super().__init__()
        self.constant_pred_class = constant_pred_class
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        batch_size = x.size(0)
        out = torch.zeros((batch_size, 2), dtype=torch.float32)
        out[:, self.constant_pred_class] = 5.0
        out[:, 1 - self.constant_pred_class] = -5.0
        return out


class TestVerifierTrainingHygiene(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name).resolve()

        # Create valid dummy crop images
        self.img_dir = self.root / "crops"
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.valid_img1 = self.img_dir / "crop1.jpg"
        self.valid_img2 = self.img_dir / "crop2.jpg"
        self.valid_img3 = self.img_dir / "crop3.jpg"

        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(self.valid_img1)
        Image.new("RGB", (64, 64), color=(150, 150, 150)).save(self.valid_img2)
        Image.new("RGB", (64, 64), color=(200, 200, 200)).save(self.valid_img3)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_csv(self, filename: str, rows: list[dict]) -> Path:
        p = self.root / filename
        fieldnames = list(rows[0].keys())
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_refuse_full_frame_image_inventory(self):
        """Verifier loader must refuse non-crop inventories (preventing full-frame dataset fallback)."""
        csv_path = self._write_csv("raw_image_inventory.csv", [
            {"crop_filepath": str(self.valid_img1), "label": "1", "split": "train"}
        ])
        with self.assertRaises(ValueError) as ctx:
            load_dataset_splits(csv_path)
        self.assertIn("Refusing to train verifier on full-frame image inventory", str(ctx.exception))

    def test_load_dataset_splits_strict_isolation(self):
        """Train set must contain only 'train', val set only 'val'; test and pilot must be excluded."""
        rows = [
            {"crop_id": "1", "crop_filepath": str(self.valid_img1), "label": "1", "split": "train", "sequence_id": "seq_A"},
            {"crop_id": "2", "crop_filepath": str(self.valid_img2), "label": "0", "split": "val", "sequence_id": "seq_B"},
            {"crop_id": "3", "crop_filepath": str(self.valid_img3), "label": "1", "split": "test", "sequence_id": "seq_C"},
            {"crop_id": "4", "crop_filepath": str(self.valid_img1), "label": "0", "split": "pilot", "sequence_id": "seq_D"},
            {"crop_id": "5", "crop_filepath": str(self.valid_img2), "label": "1", "split": "", "sequence_id": "seq_E"},
        ]
        csv_path = self._write_csv("crop_inventory.csv", rows)

        train_s, val_s, excluded_s = load_dataset_splits(csv_path)

        self.assertEqual(len(train_s), 1)
        self.assertEqual(train_s[0]["sequence_id"], "seq_A")

        self.assertEqual(len(val_s), 1)
        self.assertEqual(val_s[0]["sequence_id"], "seq_B")

        self.assertEqual(len(excluded_s), 3)
        excluded_splits = {s.get("split") for s in excluded_s}
        self.assertEqual(excluded_splits, {"test", "pilot", ""})

    def test_missing_or_corrupt_crop_raises_error_without_skip(self):
        """Dataset must raise FileNotFoundError when an image is missing, rather than substituting blank zeros."""
        missing_path = self.img_dir / "non_existent.jpg"
        samples = [{"crop_filepath": str(missing_path), "label": "1"}]

        with self.assertRaises(FileNotFoundError) as ctx:
            PotholeCropDataset(samples, allow_skip_invalid=False)
        self.assertIn("Crop image file does not exist", str(ctx.exception))

    def test_corrupt_crop_skipped_when_opt_in(self):
        """When allow_skip_invalid=True, corrupt/missing samples are excluded and tracked."""
        missing_path = self.img_dir / "non_existent.jpg"
        samples = [
            {"crop_filepath": str(self.valid_img1), "label": "1"},
            {"crop_filepath": str(missing_path), "label": "1"},
        ]
        ds = PotholeCropDataset(samples, allow_skip_invalid=True)
        self.assertEqual(len(ds), 1)
        self.assertEqual(len(ds.skipped_samples), 1)

    def test_unknown_label_raises_value_error(self):
        """Labels other than 0 and 1 (or 'pothole'/'non_pothole') must be rejected."""
        samples = [{"crop_filepath": str(self.valid_img1), "label": "maybe_pothole"}]
        with self.assertRaises(ValueError) as ctx:
            PotholeCropDataset(samples)
        self.assertIn("Invalid class label", str(ctx.exception))

    def test_check_split_independence_detects_leakage(self):
        """Overlapping sequence_id between train and val must raise ValueError."""
        train_s = [{"sequence_id": "seq_road_101", "label": "1"}]
        val_s = [{"sequence_id": "seq_road_101", "label": "0"}]

        with self.assertRaises(ValueError) as ctx:
            check_split_independence(train_s, val_s)
        self.assertIn("Data leakage detected", str(ctx.exception))
        self.assertIn("seq_road_101", str(ctx.exception))

    def test_check_split_independence_reports_clean_splits(self):
        """Independent sequence IDs must pass independence check without error."""
        train_s = [{"sequence_id": "seq_road_101", "label": "1"}]
        val_s = [{"sequence_id": "seq_road_102", "label": "0"}]

        report = check_split_independence(train_s, val_s)
        self.assertTrue(report["checked"])
        self.assertFalse(report["overlap_detected"])
        self.assertEqual(len(report["overlapping_groups"]), 0)

    def test_evaluate_verifier_computes_f1_and_confusion_matrix(self):
        """evaluate_verifier must return loss, accuracy, precision, recall, F1, and confusion matrix."""
        samples = [
            {"crop_filepath": str(self.valid_img1), "label": "1"},
            {"crop_filepath": str(self.valid_img2), "label": "0"},
        ]
        ds = PotholeCropDataset(samples)
        loader = DataLoader(ds, batch_size=2)
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        model = DummyModel(constant_pred_class=1)

        metrics = evaluate_verifier(model, loader, criterion, device)

        self.assertIn("val_f1", metrics)
        self.assertIn("val_precision", metrics)
        self.assertIn("val_recall", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["val_recall"], 1.0)
        self.assertEqual(metrics["val_precision"], 0.5)
        self.assertAlmostEqual(metrics["val_f1"], 2 * 0.5 * 1.0 / (0.5 + 1.0), places=3)
        self.assertEqual(metrics["confusion_matrix"], [[0, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
