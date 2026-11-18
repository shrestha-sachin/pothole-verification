"""Exercise ordinary staff workflows against an isolated dashboard workspace."""

from __future__ import annotations

import re
import shutil
import unittest
import uuid
from pathlib import Path

from streamlit.testing.v1 import AppTest

from dashboard.preferences import load_settings, save_settings
from src.database import Database
from src.tracking.event_grouper import CandidateEvent


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDashboardUsability(unittest.TestCase):
    def setUp(self):
        # tempfile creates directories with permissions unavailable to this
        # Windows runner. Ordinary workspace directories work and stay isolated.
        self.workspace = PROJECT_ROOT / "results" / f"test_dashboard_usability_{uuid.uuid4().hex}"
        self.workspace.mkdir(parents=True)
        self.addCleanup(self.cleanup_workspace)
        self.database_path = self.workspace / "findings.db"
        self.settings_path = self.workspace / "settings.json"
        self.model_path = self.workspace / "fake-model.pt"
        self.model_path.write_bytes(b"No model is loaded in UI tests.")
        save_settings(self.settings_path, {"detector_path": str(self.model_path)})
        self.db = Database(self.database_path)
        for video in ("video_a", "video_b"):
            self.db.register_video(video, {"filename": video + ".mp4", "filepath": "",
                                          "road_segment_id": "Main Street", "duration_sec": 30,
                                          "fps": 30, "width": 640, "height": 480})
        self.db.save_events([
            self.finding("a1", "video_a", 1000), self.finding("a2", "video_a", 2000),
            self.finding("a3", "video_a", 3000), self.finding("b1", "video_b", 1000),
        ])
        self.db.save_review("a3", "confirmed", None, "Existing confirmed finding")
        self.db.save_review("b1", "rejected", "shadow", "Existing false alarm")
        source = (PROJECT_ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        replacements = {
            "PROJECT_ROOT": PROJECT_ROOT, "DB_PATH": self.database_path,
            "SETTINGS_PATH": self.settings_path, "EVIDENCE_ROOT": self.workspace / "evidence",
            "ANNOTATED_VIDEOS_DIR": self.workspace / "marked_videos",
            "LOCAL_VIDEO_DIR": self.workspace / "source_videos",
            "MODELS_DIR": self.workspace / "models",
        }
        for name, path in replacements.items():
            source, matches = re.subn(rf"^{name} = .+$", lambda _: f"{name} = Path({str(path)!r})",
                                      source, count=1, flags=re.MULTILINE)
            self.assertEqual(matches, 1, f"Could not isolate {name}")
        start = source.index("@st.cache_data(show_spinner=False, max_entries=8)")
        end = source.index("@st.cache_data(show_spinner=False, max_entries=16)", start)
        self.source = source[:start] + (
            "def inspect_checkpoint(path, modified_ns, size):\n"
            "    return {0: 'pothole'}\n\n\n"
        ) + source[end:]
        self.script_path = self.workspace / "isolated_dashboard.py"
        self.script_path.write_text(self.source, encoding="utf-8")
        self.app = self.new_app()

    @staticmethod
    def finding(event_id, video_id, timestamp):
        return CandidateEvent(
            event_id=event_id, video_id=video_id, road_segment_id="Main Street", track_id=timestamp,
            first_timestamp_ms=timestamp, last_timestamp_ms=timestamp + 100,
            first_frame_index=1, last_frame_index=2,
            detector_conf_max=.8, detector_conf_mean=.8, observation_count=2,
            rep_x1=1, rep_y1=1, rep_x2=10, rep_y2=10, rep_timestamp_ms=timestamp,
        )

    def cleanup_workspace(self):
        # Only remove this test's verified child directory inside the workspace.
        resolved = self.workspace.resolve()
        self.assertEqual(resolved.parent, (PROJECT_ROOT / "results").resolve())
        self.assertTrue(resolved.name.startswith("test_dashboard_usability_"))
        shutil.rmtree(resolved)

    def new_app(self):
        # from_string writes into Streamlit's restrictive tempfile directory;
        # use the identical rewritten source in this ordinary workspace folder.
        app = AppTest.from_file(str(self.script_path), default_timeout=30).run()
        self.assert_no_errors(app)
        return app

    def assert_no_errors(self, app=None):
        app = app or self.app
        self.assertFalse(app.exception, [item.message for item in app.exception])

    def navigate(self, page):
        self.app.radio(key="nav_page").set_value(page).run()
        self.assert_no_errors()

    def button(self, label):
        return next(button for button in self.app.button if button.label == label)

    def test_default_navigation_and_analysis_require_no_scientific_controls(self):
        options = self.app.radio(key="nav_page").options
        self.assertIn("Settings", options)
        self.assertNotIn("Research evaluation", options)
        self.navigate("Process Video")
        self.assertEqual(self.app.radio(key="proc_source").value, "Upload a video")
        self.assertEqual(self.app.radio(key="proc_output").value, "Find potholes to review")
        self.assertEqual(len(self.app.slider), 0)
        self.assertEqual(len(self.app.number_input), 0)
        self.assertFalse(any("model" in widget.label.lower() for widget in self.app.text_input))
        self.assertTrue(self.button("Analyze video").disabled)

    def test_report_defaults_to_confirmed_findings(self):
        self.navigate("Export")
        self.assertEqual(self.app.multiselect(key="report_decisions").value, ["confirmed"])
        preview = self.app.dataframe[0].value
        self.assertEqual(list(preview["Finding ID"]), ["a3"])
        self.assertNotIn("detector_conf_max", preview.columns)

    def test_missing_review_decision_does_not_create_a_review(self):
        self.navigate("Review Events")
        self.app.selectbox(key="rev_filter_vid").set_value("video_a").run()
        self.assertEqual(self.app.selectbox(key="selected_event_id").value, "a1")
        self.button("Save decision").click().run()
        self.assert_no_errors()
        self.assertIsNone(self.db.get_event("a1")["review"])
        self.assertTrue(any("Choose a decision" in item.value for item in self.app.warning))

    def test_save_and_next_keeps_video_filter_and_persists_review(self):
        self.navigate("Review Events")
        self.app.selectbox(key="rev_filter_vid").set_value("video_a").run()
        self.app.selectbox(key="widget_dec_a1").set_value("confirmed")
        self.app.text_area(key="widget_notes_a1").set_value("Clear pothole in road")
        self.button("Save and review next").click().run()
        self.assert_no_errors()
        review = self.db.get_event("a1")["review"]
        self.assertEqual(review["decision"], "confirmed")
        self.assertEqual(review["reviewer_notes"], "Clear pothole in road")
        self.assertEqual(self.app.selectbox(key="rev_filter_vid").value, "video_a")
        self.assertEqual(self.app.selectbox(key="selected_event_id").value, "a2")
        self.assertEqual(self.db.get_event("b1")["review"]["decision"], "rejected")

    def test_saved_settings_survive_navigation_and_a_new_session(self):
        self.navigate("Settings")
        next(item for item in self.app.checkbox if item.label == "Show research tools").set_value(True)
        self.button("Save settings").click().run()
        self.assert_no_errors()
        self.assertTrue(load_settings(self.settings_path)["research_tools"])
        self.navigate("Process Video")
        self.navigate("Settings")
        self.assertTrue(next(item for item in self.app.checkbox if item.label == "Show research tools").value)
        fresh = self.new_app()
        self.assertIn("Research evaluation", fresh.radio(key="nav_page").options)

    def test_invalid_settings_leave_previous_configuration_unchanged(self):
        before = self.settings_path.read_bytes()
        self.navigate("Settings")
        next(item for item in self.app.number_input if item.label == "Repeated sightings required").set_value(6)
        next(item for item in self.app.number_input if item.label == "Number of checked images to compare").set_value(2)
        self.button("Save settings").click().run()
        self.assert_no_errors()
        self.assertTrue(self.app.error)
        self.assertEqual(self.settings_path.read_bytes(), before)

    def test_settings_accept_valid_integer_json_values_for_numeric_controls(self):
        configured = load_settings(self.settings_path)
        configured["sample_fps"] = 5
        save_settings(self.settings_path, configured)
        self.navigate("Settings")
        control = next(item for item in self.app.slider if item.label == "Video images checked per second")
        self.assertEqual(control.value, 5)

    def test_display_settings_can_be_saved_when_existing_model_is_unavailable(self):
        self.model_path.unlink()
        self.navigate("Settings")
        next(item for item in self.app.checkbox if item.label == "Show research tools").set_value(True)
        self.button("Save settings").click().run()
        self.assert_no_errors()
        self.assertTrue(load_settings(self.settings_path)["research_tools"])
        self.assertIn("Research evaluation", self.app.radio(key="nav_page").options)
