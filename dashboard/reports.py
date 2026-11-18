"""Plain-language reports for staff reviewing road footage."""
from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable, Mapping
from typing import Any


REVIEW_LABELS = {
    "confirmed": "Confirmed pothole finding",
    "rejected": "Not a pothole",
    "uncertain": "Unsure",
    "flag_grouping": "Possible duplicate or grouping issue",
    "unreviewed": "Not reviewed",
}

REPORT_COLUMNS = (
    "Video", "Road / area", "Time in video", "Review result", "Notes",
    "Reviewed at", "Finding ID",
)


def _spreadsheet_text(value: Any) -> str:
    """Keep user-entered text from being interpreted as a spreadsheet formula."""
    text = "" if value is None else str(value)
    probe = text.lstrip(" \t\r\n\x00\ufeff")
    if probe.startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")):
        return "'" + text
    return text


def _video_name(video: Mapping[str, Any], fallback: str) -> str:
    # Sidecars may come from Windows or macOS, regardless of the current host.
    value = video.get("filename") or video.get("filepath") or fallback
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _timestamp(milliseconds: Any) -> str:
    if milliseconds is None:
        return "Not recorded"
    try:
        value = float(milliseconds)
        if not math.isfinite(value) or value < 0:
            return "Not recorded"
        total_ms = int(value)
    except (ValueError, TypeError, OverflowError):
        return "Not recorded"
    seconds, ms = divmod(total_ms, 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def build_report_csv(events: Iterable[Mapping[str, Any]],
                     videos: Iterable[Mapping[str, Any]]) -> str:
    """Export exactly the supplied findings, preserving their stable IDs."""
    video_map = {v["video_id"]: v for v in videos}
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=REPORT_COLUMNS)
    writer.writeheader()
    for event in events:
        video_id = str(event.get("video_id") or "")
        video = video_map.get(video_id, {})
        review = event.get("review") or {}
        decision = review.get("decision", "unreviewed")
        road = event.get("road_segment_id") or video.get("road_segment_id")
        # The legacy placeholder is an internal default, not a recorded road name.
        if road in (None, "", "seg_001"):
            road = "Not recorded"
        timestamp = event.get("rep_timestamp_ms")
        if timestamp is None:
            timestamp = event.get("first_timestamp_ms")
        row = {
            "Video": _video_name(video, video_id),
            "Road / area": road,
            "Time in video": _timestamp(timestamp),
            "Review result": REVIEW_LABELS.get(decision, "Review status unavailable"),
            "Notes": review.get("reviewer_notes") or "",
            "Reviewed at": review.get("reviewed_at") or "",
            "Finding ID": event.get("event_id") or "",
        }
        writer.writerow({key: _spreadsheet_text(value) for key, value in row.items()})
    return out.getvalue()


def _safe_technical_csv(csv_text: str) -> str:
    """Protect text cells in the existing technical export without changing scores."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return csv_text
    text_columns = {
        "event_id", "video_id", "road_segment_id", "full_frame_path", "crop_path", "clip_path",
        "review_decision", "fp_category", "reviewer_notes", "reviewed_at", "model_path",
        "model_version", "run_config", "created_at",
    }
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        writer.writerow({key: _spreadsheet_text(value) if key in text_columns else value
                         for key, value in row.items()})
    return out.getvalue()


def build_crop_manifest_csv(events: Iterable[Mapping[str, Any]]) -> str:
    """List saved training-crop references without opening or creating any files."""
    columns = ("event_id", "video_id", "label", "subtype", "crop_path",
               "detector_conf_max", "rep_timestamp_ms")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    for event in events:
        review = event.get("review") or {}
        decision = review.get("decision")
        if decision not in ("confirmed", "rejected") or not event.get("crop_path"):
            continue
        positive = decision == "confirmed"
        row = {
            "event_id": event.get("event_id") or "",
            "video_id": event.get("video_id") or "",
            "label": "pothole" if positive else "non_pothole",
            "subtype": "pothole" if positive else review.get("fp_category") or "ordinary_road",
            "crop_path": event["crop_path"],
            "detector_conf_max": event.get("detector_conf_max"),
            "rep_timestamp_ms": event.get("rep_timestamp_ms", event.get("first_timestamp_ms")),
        }
        writer.writerow({key: _spreadsheet_text(value) if key not in
                         ("detector_conf_max", "rep_timestamp_ms") else value
                         for key, value in row.items()})
    return out.getvalue()


def render_reports(db: Any, research_tools: bool = False) -> None:
    """Render normal reporting first; make technical exports an explicit opt-in."""
    import streamlit as st

    st.subheader("Reports")
    st.caption("Download a spreadsheet of reviewed findings to share with your team.")
    videos = db.get_videos()
    video_map = {v["video_id"]: v for v in videos}
    left, right = st.columns([1, 2])
    with left:
        scope = st.selectbox(
            "Video", [None] + list(video_map), key="report_video",
            format_func=lambda value: "All videos" if value is None else _video_name(video_map[value], value),
        )
    with right:
        decisions = st.multiselect(
            "Include findings with these review results",
            list(REVIEW_LABELS), default=["confirmed"],
            format_func=REVIEW_LABELS.get, key="report_decisions",
            help="Only findings confirmed by a person are included by default.",
        )
    events = db.get_events(video_id=scope, include_reviews=True)
    selected = [event for event in events
                if (event.get("review") or {}).get("decision", "unreviewed") in decisions]
    if not decisions:
        st.info("Choose at least one review result to create a report.")
    elif not events:
        st.info("No findings are available for this selection. Analyze a video using 'Find potholes to review' first.")
    elif not selected:
        st.info("No findings match these review results. Review possible potholes or include another result above.")
    else:
        st.write(f"**{len(selected):,} findings** included in this report.")
        csv_text = build_report_csv(selected, videos)
        st.download_button(
            "Download report (CSV)", data=csv_text.encode("utf-8-sig"),
            file_name="road_findings_report.csv", mime="text/csv",
            type="primary", icon=":material/download:", on_click="ignore",
        )
        st.dataframe(list(csv.DictReader(io.StringIO(csv_text)))[:25],
                     width="stretch", hide_index=True)
        if len(selected) > 25:
            st.caption("Showing the first 25 findings. The download includes all matching findings.")

    st.caption(
        "A finding is an area flagged in a video. Different findings may show the same physical defect. "
        "No geographic location is recorded; use the video timestamp to find the area in the footage. "
        "Review dates include their saved time zone when available."
    )

    if research_tools:
        with st.expander("Technical exports"):
            st.caption("Full analysis records include model scores and internal file paths for research or audit work.")
            st.caption("These exports include all review results for the selected video, regardless of the review filter above.")
            if st.checkbox("Prepare full analysis CSV", key="report_prepare_technical"):
                st.download_button(
                    "Download full analysis CSV", _safe_technical_csv(db.export_events_csv(video_id=scope)).encode("utf-8-sig"),
                    "full_analysis.csv", "text/csv", on_click="ignore",
                )
            if st.checkbox("Prepare previous run archive", key="report_prepare_archive"):
                st.download_button(
                    "Download previous run archive", db.export_run_archive(video_id=scope),
                    "run_archive.json", "application/json", on_click="ignore",
                )
            if st.checkbox("Prepare training crop inventory", key="report_prepare_crops"):
                st.caption("Lists saved image paths for confirmed and rejected findings. Images are not bundled, and paths may refer to files no longer available.")
                crop_csv = build_crop_manifest_csv(events)
                crop_rows = list(csv.DictReader(io.StringIO(crop_csv)))
                if crop_rows:
                    st.caption(f"{len(crop_rows):,} labeled crop references included.")
                    st.download_button(
                        "Download crop manifest CSV", crop_csv.encode("utf-8-sig"),
                        "reviewed_crop_inventory.csv", "text/csv", on_click="ignore",
                    )
                else:
                    st.info("No confirmed or rejected findings with saved crop paths are available for this video selection.")
