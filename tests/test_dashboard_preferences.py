"""Durable settings survive page navigation and reject invalid configuration."""

import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from dashboard.preferences import DEFAULT_SETTINGS, load_settings, save_settings, validate_settings


class TestDashboardPreferences(unittest.TestCase):
    def test_missing_and_broken_files_return_independent_defaults(self):
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            result = load_settings(Path("missing.json"))
        self.assertEqual(result, DEFAULT_SETTINGS)
        result["tracking"] = False
        self.assertTrue(DEFAULT_SETTINGS["tracking"])
        for content in ("{broken", "[]", '{"sample_fps": "5"}', '{"tracking": null}'):
            with self.subTest(content=content), patch.object(Path, "read_text", return_value=content):
                self.assertEqual(load_settings(Path("broken.json")), DEFAULT_SETTINGS)

    def test_partial_configuration_merges_defaults_and_ignores_unknown_keys(self):
        with patch.object(Path, "read_text", return_value=json.dumps({"sample_fps": 8, "future": 1})):
            loaded = load_settings(Path("saved.json"))
        self.assertEqual(loaded["sample_fps"], 8)
        self.assertEqual(loaded["detector_path"], DEFAULT_SETTINGS["detector_path"])
        self.assertNotIn("future", loaded)

    def test_validation_rejects_wrong_types_nonfinite_and_invalid_ranges(self):
        invalid = [
            {"detector_conf": True}, {"detector_conf": "0.35"},
            {"detector_iou": float("nan")}, {"sample_fps": float("inf")},
            {"sample_fps": 0}, {"sample_fps": 10 ** 400}, {"verifier_threshold": 1}, {"imgsz": 641},
            {"imgsz": 640.0}, {"min_obs": 2.0}, {"temporal_window": True},
            {"min_obs": 6, "temporal_window": 5}, {"tracking": 1},
            {"generate_clips": "false"}, {"research_tools": None}, {"detector_path": 4},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_settings(values)

    def test_persistence_round_trip_and_failed_write_preserve_previous_snapshot(self):
        path = Path(__file__).resolve().parent.parent / "results" / f"test_preferences_{uuid.uuid4().hex}.json"
        try:
            configured = validate_settings({"research_tools": True, "sample_fps": 7.5})
            save_settings(path, configured)
            self.assertEqual(load_settings(path), configured)
            with patch.object(Path, "replace", side_effect=OSError("simulated write failure")):
                with self.assertRaises(OSError):
                    save_settings(path, {"tracking": False})
            self.assertEqual(load_settings(path), configured)
            self.assertFalse(list(path.parent.glob(f".{path.name}.*.part")))
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_save_does_not_touch_filesystem(self):
        with patch.object(Path, "mkdir") as mkdir, self.assertRaises(ValueError):
            save_settings(Path("never-created/settings.json"), {"tracking": "yes"})
        mkdir.assert_not_called()
