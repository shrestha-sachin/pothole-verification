"""Regression tests for staff reports and safe spreadsheet text."""
import csv
import io
import unittest
from unittest.mock import MagicMock, patch

from dashboard.reports import REPORT_COLUMNS, _safe_technical_csv, build_crop_manifest_csv, build_report_csv, render_reports


class TestOperationalReport(unittest.TestCase):
    def read_rows(self, events, videos=()):
        return list(csv.DictReader(io.StringIO(build_report_csv(events, videos))))

    def test_report_has_readable_fields_and_preserves_finding_identity(self):
        event = {
            "event_id": "ev-0007", "video_id": "vid1", "road_segment_id": "North Avenue",
            "rep_timestamp_ms": 62015,
            "review": {"decision": "confirmed", "reviewer_notes": 'Needs a "closer look", near curb',
                       "reviewed_at": "2026-11-18T12:00:00+00:00"},
        }
        video = {"video_id": "vid1", "filepath": "C:\\private\\road_video.mp4"}
        row = self.read_rows([event], [video])[0]
        self.assertEqual(tuple(row), REPORT_COLUMNS)
        self.assertEqual(row["Finding ID"], "ev-0007")
        self.assertEqual(row["Video"], "road_video.mp4")
        self.assertEqual(row["Road / area"], "North Avenue")
        self.assertEqual(row["Time in video"], "00:01:02.015")
        self.assertEqual(row["Review result"], "Confirmed pothole finding")
        self.assertEqual(row["Notes"], 'Needs a "closer look", near curb')
        self.assertEqual(row["Reviewed at"], "2026-11-18T12:00:00+00:00")
        self.assertNotIn("model_path", row)

    def test_user_text_cannot_start_a_spreadsheet_formula(self):
        for dangerous in ("=1+1", "+SUM(A1)", "-1+1", "@SUM(A1)", "  =2", "\t=3", "\r=3", "\n=3"):
            with self.subTest(dangerous=dangerous):
                event = {"event_id": "safe-id", "video_id": "v1", "road_segment_id": dangerous,
                         "review": {"decision": "confirmed", "reviewer_notes": dangerous}}
                row = self.read_rows([event], [{"video_id": "v1", "filename": dangerous}])[0]
                for column in ("Road / area", "Notes", "Video"):
                    self.assertTrue(row[column].startswith("'"), column)
                self.assertEqual(row["Finding ID"], "safe-id")

    def test_missing_metadata_and_unreviewed_findings_are_explicit(self):
        row = self.read_rows([{"event_id": "e1", "video_id": "orphan", "road_segment_id": "seg_001"}])[0]
        self.assertEqual(row["Video"], "orphan")
        self.assertEqual(row["Road / area"], "Not recorded")
        self.assertEqual(row["Time in video"], "Not recorded")
        self.assertEqual(row["Review result"], "Not reviewed")
        self.assertEqual(row["Notes"], "")

    def test_timestamp_fallback_and_long_videos(self):
        rows = self.read_rows([
            {"event_id": "e1", "first_timestamp_ms": 3661002},
            {"event_id": "e2", "rep_timestamp_ms": None, "first_timestamp_ms": 0},
            {"event_id": "e3", "rep_timestamp_ms": float("nan")},
            {"event_id": "e4", "rep_timestamp_ms": -1},
        ])
        self.assertEqual([row["Time in video"] for row in rows],
                         ["01:01:01.002", "00:00:00.000", "Not recorded", "Not recorded"])

    def test_all_requested_decisions_are_preserved_without_filtering(self):
        rows = self.read_rows([
            {"event_id": "a", "review": {"decision": "confirmed"}},
            {"event_id": "b", "review": {"decision": "rejected"}},
            {"event_id": "c", "review": {"decision": "uncertain"}},
            {"event_id": "d", "review": {"decision": "flag_grouping"}},
        ])
        self.assertEqual([row["Finding ID"] for row in rows], ["a", "b", "c", "d"])
        self.assertEqual([row["Review result"] for row in rows], [
            "Confirmed pothole finding", "Not a pothole", "Unsure", "Possible duplicate or grouping issue"])

    def test_empty_report_still_has_headers(self):
        csv_text = build_report_csv([], [])
        reader = csv.DictReader(io.StringIO(csv_text))
        self.assertEqual(reader.fieldnames, list(REPORT_COLUMNS))
        self.assertEqual(list(reader), [])

    def test_technical_export_protects_notes_and_preserves_numeric_values(self):
        source = "event_id,reviewer_notes,rep_x1\r\ne1,=1+1,-4\r\n"
        row = next(csv.DictReader(io.StringIO(_safe_technical_csv(source))))
        self.assertEqual(row["reviewer_notes"], "'=1+1")
        self.assertEqual(row["rep_x1"], "-4")

    def test_training_manifest_keeps_only_labeled_crop_references(self):
        events = [
            {"event_id": "yes", "video_id": "v1", "crop_path": "missing/yes.jpg",
             "review": {"decision": "confirmed"}},
            {"event_id": "no", "video_id": "v1", "crop_path": "missing/no.jpg",
             "review": {"decision": "rejected", "fp_category": "shadow"}},
            {"event_id": "unsure", "crop_path": "missing/unsure.jpg", "review": {"decision": "uncertain"}},
            {"event_id": "empty", "review": {"decision": "confirmed"}},
            {"event_id": "pending", "crop_path": "missing/pending.jpg"},
        ]
        with patch("pathlib.Path.open", side_effect=AssertionError("Must not open image files")):
            rows = list(csv.DictReader(io.StringIO(build_crop_manifest_csv(events))))
        self.assertEqual([row["event_id"] for row in rows], ["yes", "no"])
        self.assertEqual([row["label"] for row in rows], ["pothole", "non_pothole"])
        self.assertEqual(rows[1]["subtype"], "shadow")

    def test_training_manifest_protects_free_text(self):
        events = [{"event_id": "e1", "video_id": "=2", "crop_path": "@bad",
                   "review": {"decision": "rejected", "fp_category": "=1+1"}}]
        row = next(csv.DictReader(io.StringIO(build_crop_manifest_csv(events))))
        self.assertEqual(row["video_id"], "'=2")
        self.assertEqual(row["crop_path"], "'@bad")
        self.assertEqual(row["subtype"], "'=1+1")

    def test_run_archive_respects_selected_video_and_other_exports_stay_lazy(self):
        fake_st = MagicMock()
        fake_st.columns.return_value = [MagicMock(), MagicMock()]
        fake_st.selectbox.return_value = "v1"
        fake_st.multiselect.return_value = ["confirmed"]
        fake_st.checkbox.side_effect = lambda *args, **kwargs: kwargs.get("key") == "report_prepare_archive"
        db = MagicMock()
        db.get_videos.return_value = [{"video_id": "v1", "filename": "one.mp4"}]
        db.get_events.return_value = []
        db.export_run_archive.return_value = "[]"
        with patch.dict("sys.modules", {"streamlit": fake_st}):
            render_reports(db, research_tools=True)
        db.export_run_archive.assert_called_once_with(video_id="v1")
        db.export_events_csv.assert_not_called()
        db.get_events.assert_called_once_with(video_id="v1", include_reviews=True)


if __name__ == "__main__":
    unittest.main()
