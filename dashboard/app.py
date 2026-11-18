"""RoadLens: analyze road footage, review findings, watch videos, and report results.

Launch: streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Project root & sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.preferences import load_settings, save_settings, validate_settings
from dashboard.reports import render_reports

st.set_page_config(
    page_title="RoadLens | Pothole Review",
    page_icon=":material/road:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Shared layout styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Use system typography: no remote font request or fragile generated selectors. */
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
/* Callout Box */
.info-callout {
    background: rgba(37, 99, 235, 0.06);
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.85rem;
    margin: 10px 0 16px;
    color: inherit;
}
.info-callout.warning {
    background: rgba(217, 119, 6, 0.07);
    border-left-color: #d97706;
}
.info-callout.success {
    background: rgba(22, 163, 74, 0.07);
    border-left-color: #16a34a;
}
.info-callout.error {
    background: rgba(220, 38, 38, 0.07);
    border-left-color: #dc2626;
}

/* Section Header */
.section-hdr {
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.25);
    margin: 18px 0 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dependencies & Imports
# ---------------------------------------------------------------------------
try:
    import cv2
    import pandas as pd
except ImportError as e:
    st.error(f"Missing core dependency: {e}. Run: `pip install opencv-python numpy pandas`")
    st.stop()

try:
    from src.database import Database
    from src.detector.video_sampler import iter_frames, get_video_metadata
    from src.tracking.event_grouper import EventGrouper
    from src.evaluation.policies import PolicyConfig, apply_policies
    from src.verifier.interface import load_verifier
    from src.evaluation.matcher import (
        GroundTruthAnnotation,
        evaluate_event_detections,
        run_sensitivity_analysis,
        load_video_ground_truth,
        parse_ground_truth_csv,
    )
    from src.video.review_metadata import normalize_review_metadata
    from src.video.annotated_renderer import (
        AnnotatedVideoRenderer,
    )
except ImportError as e:
    st.error(f"Project import failure: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Paths & Setup
# ---------------------------------------------------------------------------
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "evidence"
DB_PATH = PROJECT_ROOT / "results" / "pothole_verification.db"
LOCAL_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "local_video"
MODELS_DIR = PROJECT_ROOT / "models"
ANNOTATED_VIDEOS_DIR = PROJECT_ROOT / "results" / "annotated_videos"
SETTINGS_PATH = PROJECT_ROOT / "configs" / "dashboard_settings.json"

EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
ANNOTATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=8)
def inspect_checkpoint(path: str, modified_ns: int, size: int):
    """Cache only class labels; never retain the full model for UI validation."""
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return getattr(checkpoint.get("model"), "names", None) or checkpoint.get("names")
    return None


@st.cache_data(show_spinner=False, max_entries=16)
def video_preview(path: str, modified_ns: int, size: int):
    metadata = get_video_metadata(path)
    cap = cv2.VideoCapture(path)
    try:
        ok, frame = cap.read()
        preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()
    return metadata, preview


def get_db() -> Database:
    db = st.session_state.get("db")
    if db is None or not hasattr(db, "get_annotated_videos"):
        try:
            import importlib
            import src.database
            importlib.reload(src.database)
            from src.database import Database
            st.session_state["db"] = Database(DB_PATH)
        except Exception:
            from src.database import Database
            st.session_state["db"] = Database(DB_PATH)
    return st.session_state["db"]


def load_annotated_video_records(db: Database, video_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Load annotated video records from database and disk metadata cache."""
    records: list[dict[str, Any]] = []
    if hasattr(db, "get_annotated_videos"):
        try:
            records = db.get_annotated_videos(video_id=video_id)
        except Exception as ex:
            logger.warning("Error fetching annotated videos from database: %s", ex)
            records = []

    # Also merge from results/annotated_videos/*.json as safe disk discovery
    known_job_ids = {r.get("job_id") for r in records if r.get("job_id")}
    if ANNOTATED_VIDEOS_DIR.exists():
        for json_file in sorted(ANNOTATED_VIDEOS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(json_file.read_text())
                jid = meta.get("job_id")
                vid = meta.get("video_id")
                if video_id and vid != video_id:
                    continue
                if jid and jid not in known_job_ids:
                    out_path = Path(meta.get("output_video_path", ""))
                    if out_path.exists():
                        records.append(meta)
                        known_job_ids.add(jid)
            except Exception:
                continue
    return sorted((normalize_review_metadata(r) for r in records if r.get("status", "completed") == "completed"), key=lambda r: r.get("created_at", ""), reverse=True)


def init_session_state():
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview"
    if "selected_event_id" not in st.session_state:
        st.session_state["selected_event_id"] = None
    if "review_saved_msg" not in st.session_state:
        st.session_state["review_saved_msg"] = None


init_session_state()


def fmt_ts(ms: int | float) -> str:
    """Format milliseconds as MM:SS."""
    s = int(ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def fmt_ts_full(ms: int | float) -> str:
    """Format milliseconds as MM:SS.mmm."""
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}.{int(ms) % 1000:03d}"


def auto_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def go_to(page: str):
    st.session_state["pending_nav_page"] = page
    st.rerun()


DECISION_LABELS = {
    "unreviewed": "Needs review", "confirmed": "Pothole",
    "rejected": "Not a pothole", "uncertain": "Unsure",
    "flag_grouping": "Possible duplicate or grouping issue",
}
REASON_LABELS = {"": "Not specified", "shadow": "Shadow", "repaired_patch": "Repaired road",
                 "manhole": "Manhole cover", "ordinary_road": "Undamaged road", "other": "Other"}


def current_settings():
    return load_settings(SETTINGS_PATH)


def video_label(video: dict):
    return video.get("filename") or Path(video.get("filepath") or "").name or video["video_id"]


def road_label(value):
    return "Not specified" if value in (None, "", "seg_001") else value


def configured_detector(settings):
    path = Path(settings["detector_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        return None, "The pothole detector has not been set up. Open Settings to choose a model file."
    try:
        stat = path.stat()
        names = inspect_checkpoint(str(path), stat.st_mtime_ns, stat.st_size)
        class_zero = names.get(0, names.get("0", "")) if isinstance(names, dict) else names[0] if names else ""
        if "pothole" not in str(class_zero).lower():
            return None, "The selected model is not a compatible pothole detector. Choose a pothole model in Settings."
    except Exception:
        logger.exception("Could not validate detector")
        return None, "The detector file could not be checked. Open Settings to choose a working pothole model."
    return path, None


def page_overview():
    st.subheader("Overview", anchor=False)
    st.caption("Turn road footage into findings your team can review and report.")
    db = get_db()
    stats = db.get_summary_stats()
    videos = db.get_videos()
    recordings = load_annotated_video_records(db)
    ready_ids = {v["video_id"] for v in videos if v.get("processing_status") == "done"}
    watch_ids = {r["video_id"] for r in recordings}
    needs_review = max(0, stats["total_events"] - stats["reviewed"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos analyzed", len(ready_ids | watch_ids))
    c2.metric("Needs review", needs_review)
    c3.metric("Confirmed findings", stats["confirmed"])
    c4.metric("Unsure or flagged", stats.get("uncertain", 0) + stats.get("flag_grouping", 0))
    st.caption("Findings are areas flagged in a video. The same physical pothole may appear in more than one finding.")

    if stats["total_events"]:
        st.progress(min(1.0, stats["reviewed"] / stats["total_events"]),
                    text=f"{stats['reviewed']} of {stats['total_events']} findings reviewed")
    elif recordings:
        st.info("Your marked videos are ready to watch. To create a review list and report, analyze a video with 'Find potholes to review'.")
    else:
        st.info("Start with a road video. The app will flag possible potholes for you to check.")

    st.markdown("### What would you like to do?")
    for col, title, text, button, page in zip(
        st.columns(3),
        ["Analyze footage", "Check possible potholes", "Share the results"],
        ["Upload a road recording or choose one already on this computer.",
         "Look at each image and decide whether it shows a pothole.",
         "Download a spreadsheet of confirmed findings and your notes."],
        ["Analyze a video", "Review findings", "Download a report"],
        ["Process Video", "Review Events", "Export"],
    ):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
            if st.button(button, width="stretch", type="primary" if page == ("Review Events" if needs_review else "Process Video") else "secondary"):
                go_to(page)

    if videos:
        st.markdown("### Recent videos")
        statuses = {"done": "Ready to review", "processing": "Analyzing", "error": "Needs attention", "pending": "Not analyzed"}
        rows = [{"Video": video_label(v), "Road or area": road_label(v.get("road_segment_id")),
                 "Status": "Ready to watch" if v["video_id"] in watch_ids and v["video_id"] not in ready_ids else statuses.get(v.get("processing_status"), "Not analyzed")}
                for v in videos[:8]]
        st.dataframe(rows, width="stretch", hide_index=True)


def page_process():
    st.subheader("Analyze a video")
    st.caption("Choose road footage, then tell us how you want to use the results.")
    db = get_db()
    settings = current_settings()
    st.markdown("### 1. Choose your video")
    source = st.radio("Video source", ["Upload a video", "Choose a saved video"], horizontal=True, key="proc_source")
    video_path_final = None
    if source == "Upload a video":
        uploaded = st.file_uploader("Road video", type=["mp4", "mov", "avi"], help="Choose an MP4, MOV, or AVI recording from your device.")
        if uploaded is not None:
            # Keep the content hash and saved path across UI reruns for this upload.
            upload_key = (uploaded.file_id, uploaded.name, uploaded.size)
            saved = st.session_state.get("saved_upload")
            if not saved or saved[0] != upload_key or not Path(saved[1]).is_file():
                LOCAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                content = uploaded.getvalue()
                name = Path(uploaded.name)
                destination = LOCAL_VIDEO_DIR / f"{name.stem}_{hashlib.sha256(content).hexdigest()[:12]}{name.suffix.lower()}"
                if not destination.exists():
                    destination.write_bytes(content)
                st.session_state["saved_upload"] = (upload_key, str(destination))
            video_path_final = Path(st.session_state["saved_upload"][1])
    else:
        local = sorted(p for p in LOCAL_VIDEO_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi"}) if LOCAL_VIDEO_DIR.exists() else []
        if local:
            chosen = st.selectbox("Saved video", [str(p) for p in local], format_func=lambda p: Path(p).name)
            video_path_final = Path(chosen)
        else:
            st.info("No saved videos yet. Choose 'Upload a video' to add one.")
        if settings["research_tools"]:
            with st.expander("Use another file on this computer"):
                override = st.text_input("Video file path")
                if override.strip():
                    video_path_final = Path(override.strip())
                    if not video_path_final.is_absolute():
                        video_path_final = PROJECT_ROOT / video_path_final

    video_id = video_path_final.stem.replace(" ", "_") if video_path_final else ""
    video_ready = False
    if video_path_final:
        if not video_path_final.is_file():
            st.error("The video file is unavailable. Choose another recording.")
        else:
            stat = video_path_final.stat()
            meta, preview = video_preview(str(video_path_final), stat.st_mtime_ns, stat.st_size)
            if meta.get("error") or not meta.get("duration_sec", 0) > 0:
                st.error("This video could not be opened. Try another recording or save it as an MP4.")
            else:
                video_ready = True
                st.caption(f"{video_path_final.name} | {meta['duration_sec'] / 60:.1f} minutes | {meta['file_size_mb']:.1f} MB")
                with st.expander("Preview video"):
                    if preview is not None:
                        st.image(preview, width="stretch")

    st.markdown("### 2. Choose your result")
    output = st.radio("What do you need?", ["Find potholes to review", "Create a video with marked potholes"],
                      key="proc_output", horizontal=True)
    review_output = output == "Find potholes to review"
    if review_output:
        st.caption("Creates a list of possible potholes with images and video times. Review them, then download a report.")
        road = st.text_input("Road or area", placeholder="For example: Main Street, between 1st and 5th Avenue", key="proc_road",
                             help="This name appears in your report. This video analysis does not record a GPS location.")
    else:
        road = ""
        st.caption("Creates a playable video with boxes around possible potholes. This option does not create a review list or spreadsheet report.")

    existing = db.get_events(video_id=video_id, include_reviews=True) if video_id and review_output else []
    replace_ok = True
    if existing:
        st.warning(f"This video already has {len(existing)} findings. Analyzing it again replaces the current findings. Existing reviews are matched to the new findings where possible; earlier results are archived.")
        replace_ok = st.checkbox("Analyze again and replace the current findings", key=f"replace_{video_id}")
    model_path, setup_error = configured_detector(settings)
    if setup_error:
        st.info(setup_error)
        if st.button("Open Settings", icon=":material/settings:"):
            go_to("Settings")
    else:
        st.caption("Using saved analysis settings. You can change them in Settings.")
    can_analyze = video_ready and model_path is not None and replace_ok and (bool(road.strip()) or not review_output)
    if review_output and video_ready and not road.strip():
        st.caption("Enter a road or area name to include it in your report.")
    if st.button("Analyze video", type="primary", icon=":material/play_arrow:", disabled=not can_analyze, width="stretch"):
        if review_output:
            verifier = Path(settings["verifier_path"]) if settings["verifier_path"] else None
            if verifier and not verifier.is_absolute():
                verifier = PROJECT_ROOT / verifier
            _execute_video_pipeline(
                db=db, video_path=video_path_final, video_id=video_id, model_path=model_path,
                verifier_path=str(verifier) if verifier else None, sample_fps=settings["sample_fps"],
                conf_threshold=settings["detector_conf"], iou_threshold=settings["detector_iou"], imgsz=settings["imgsz"],
                road_segment_id=road.strip(), policy_b_min_obs=settings["min_obs"], policy_b_window=settings["temporal_window"],
                verifier_threshold=settings["verifier_threshold"], preserve_reviews=True, generate_clips=settings["generate_clips"],
            )
        else:
            _execute_annotated_video_pipeline(db=db, video_path=video_path_final, video_id=video_id,
                model_path=model_path, conf_threshold=settings["detector_conf"], imgsz=settings["imgsz"],
                use_tracking=settings["tracking"], iou_threshold=settings["detector_iou"])


def _render_annotated_video_view(meta_dict: dict[str, Any]):
    """Present playback before secondary analysis information."""
    output_path = Path(meta_dict["output_video_path"])
    if not output_path.is_file():
        st.warning("This recording is no longer available. Analyze the source video again to create a new marked video.")
        return
    st.video(str(output_path))
    st.caption("Boxes mark possible potholes suggested by the software. Check the footage before treating them as confirmed.")
    _render_video_actions(meta_dict)


@st.fragment
def _render_video_actions(meta_dict: dict[str, Any]):
    """Download controls rerun independently, preserving the playing video."""
    output_path = Path(meta_dict["output_video_path"])
    # Download data is expensive: do not allocate a second full-video buffer by default.
    if st.checkbox("Prepare MP4 download", key=f"download_{meta_dict['job_id']}"):
        with output_path.open("rb") as video_file:
            st.download_button("Download MP4", data=video_file, file_name=output_path.name,
                               mime="video/mp4", on_click="ignore", icon=":material/download:")
    if current_settings()["research_tools"]:
        with st.expander("Technical analysis details"):
            st.json(meta_dict)


def page_video_library():
    st.subheader("Watch videos")
    st.caption("Watch road footage with possible potholes marked. Use the player controls to change speed or enter fullscreen.")
    records = load_annotated_video_records(get_db())
    if not records:
        st.info("No marked videos yet. In Analyze a video, choose Create a video with marked potholes.")
        return
    by_job = {r["job_id"]: r for r in records}
    selected = st.selectbox(
        "Recording", list(by_job), key="library_job",
        format_func=lambda job: (
            f"{Path(by_job[job].get('input_video_path', '')).name} | "
            f"{by_job[job].get('created_at', '')[:19].replace('T', ' ')}"
        ),
    )
    _render_annotated_video_view(by_job[selected])


def _execute_annotated_video_pipeline(
    db: Database,
    video_path: Path,
    video_id: str,
    model_path: Path,
    conf_threshold: float,
    imgsz: int,
    use_tracking: bool,
    iou_threshold: float = 0.45,
):
    from src.video.annotated_renderer import AnnotatedVideoRenderer, RenderProgress

    renderer = AnnotatedVideoRenderer(output_dir=ANNOTATED_VIDEOS_DIR)
    device = auto_device()

    with st.status("Creating your marked video...", expanded=True) as status:
        st.write("1. Preparing the pothole detector...")
        prog_bar = st.progress(0.0)
        prog_text = st.empty()

        def on_prog(p: RenderProgress):
            prog_bar.progress(p.percent / 100.0)
            eta_txt = f" | About {p.eta_sec:.0f}s remaining" if p.eta_sec is not None else ""
            prog_text.text(f"{p.percent:.0f}% complete | Elapsed: {p.elapsed_sec:.0f}s{eta_txt}")

        st.write("2. Checking the video and marking possible potholes...")
        try:
            meta = renderer.render_annotated_video(
                video_path=video_path,
                model=model_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                imgsz=imgsz,
                device=device,
                use_tracking=use_tracking,
                progress_callback=on_prog,
            )
        except Exception as exc:
            prog_bar.empty()
            prog_text.empty()
            logger.exception("Annotated video rendering failed")
            status.update(label="Video rendering failed", state="error", expanded=False)
            st.error(f"Annotated video rendering failed: {exc}", icon=":material/error:")
            return

        prog_bar.empty()
        prog_text.empty()

        st.write("3. Saving the marked video...")
        if hasattr(db, "save_annotated_video"):
            db.save_annotated_video(meta)

        # Register video in db if not already registered
        if not db.get_video(video_id):
            meta_dict = get_video_metadata(str(video_path))
            meta_dict["filepath"] = str(video_path)
            meta_dict["road_segment_id"] = ""
            db.register_video(video_id, meta_dict)

        status.update(
            label="Your marked video is ready",
            state="complete",
            expanded=False,
        )

    st.session_state[f"active_annotated_{video_id}"] = meta.job_id
    st.session_state["library_job"] = meta.job_id
    st.session_state["pending_nav_page"] = "Video Library"
    st.session_state["analysis_notice"] = "Your marked video is ready to watch. Boxes show possible potholes, not confirmed findings."
    st.rerun()


def _execute_video_pipeline(**kwargs):
    attempt_root = EVIDENCE_ROOT / "runs" / uuid.uuid4().hex
    try:
        return _run_video_pipeline(attempt_root=attempt_root, **kwargs)
    except Exception as exc:
        published = json.loads((kwargs["db"].get_video(kwargs["video_id"]) or {}).get("run_config") or "{}")
        if published.get("run_id") != attempt_root.name:
            shutil.rmtree(attempt_root, ignore_errors=True)
            message = f"Processing failed: {exc}. Previous successful results and reviews are unchanged."
        else:
            message = f"Results were saved, but the completion display failed: {exc}. Open Review findings."
        logger.exception("Video processing failed")
        st.error(message, icon=":material/error:")


def _run_video_pipeline(
    attempt_root: Path,
    db: Database,
    video_path: Path,
    video_id: str,
    model_path: Path,
    verifier_path: Optional[str],
    sample_fps: float,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    road_segment_id: str,
    policy_b_min_obs: int,
    policy_b_window: int,
    verifier_threshold: float,
    preserve_reviews: bool,
    generate_clips: bool,
):
    from ultralytics import YOLO

    meta = get_video_metadata(str(video_path))
    meta["filepath"] = str(video_path)
    meta["road_segment_id"] = road_segment_id
    if meta.get("error") or not meta.get("duration_sec", 0) > 0:
        raise ValueError(meta.get("error", "Video must report a positive duration"))
    run_config = {
        "run_id": attempt_root.name, "sample_fps": sample_fps,
        "model_path": str(model_path), "detector_conf": conf_threshold,
        "detector_iou": iou_threshold, "imgsz": imgsz,
        "min_obs": policy_b_min_obs, "temporal_window_frames": policy_b_window,
        "verifier_path": verifier_path, "verifier_threshold": verifier_threshold,
        "preprocessing_version": "context15-letterbox224-imagenet-v1",
    }
    def checkpoint_hash(path):
        if not path:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    import importlib.metadata
    run_config["packages"] = {}
    for package in ("streamlit", "ultralytics", "torch", "torchvision", "numpy"):
        try:
            run_config["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            run_config["packages"][package] = "not installed"
    run_config["detector_sha256"] = checkpoint_hash(model_path)
    run_config["verifier_sha256"] = checkpoint_hash(verifier_path)

    t0 = time.time()
    device = auto_device()

    with st.status("Analyzing your road video...", expanded=True) as status:
        st.write("1. Preparing the pothole detector...")
        model = YOLO(str(model_path))

        grouper = EventGrouper(
            video_id=video_id,
            road_segment_id=road_segment_id,
            policy_b_min_obs=policy_b_min_obs,
            policy_b_window_frames=policy_b_window,
        )
        grouper.validate_checkpoint(model)

        total_sec = meta.get("duration_sec", 0) or 0
        expected = max(1, int(total_sec * sample_fps))
        st.write("2. Checking the video for possible potholes...")

        prog_bar = st.progress(0.0)
        prog_text = st.empty()

        actual_frames = [0]
        def on_prog(fi, tot):
            actual_frames[0] = fi
            fraction = min(1.0, fi / expected)
            prog_bar.progress(fraction)
            elapsed = time.time() - t0
            prog_text.text(f"Processed frame {fi} of ~{expected} ({fraction*100:.0f}%) — Elapsed: {elapsed:.1f}s")

        frame_iter = iter_frames(str(video_path), target_fps=sample_fps)

        events = grouper.process_video_stream(
            frame_iter=frame_iter,
            model=model,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            evidence_root=attempt_root,
            video_path=video_path,
            generate_clips=generate_clips,
            total_expected_frames=expected,
            progress_callback=on_prog,
        )
        prog_bar.empty()
        prog_text.empty()
        st.write(f"Found **{len(events)} areas** to review.")

        # Verifier scoring
        verifier = load_verifier(verifier_path, threshold=verifier_threshold)
        if verifier.is_configured():
            st.write("3. Checking the images with the additional model...")
            for ev in events:
                if ev.crop_path and Path(ev.crop_path).exists():
                    cbgr = cv2.imread(ev.crop_path)
                    if cbgr is not None:
                        res = verifier.score_crop(cbgr)
                        ev.policy_c_score = res.pothole_score
                        ev.policy_c_retained = res.is_pothole
        else:
            st.write("3. No additional image-check model is set up. Continuing with the pothole detector.")
            for ev in events:
                ev.policy_c_score = None
                ev.policy_c_retained = None

        # Reporting policies
        st.write("4. Organizing findings for review...")
        pcfg = PolicyConfig(
            conf_threshold=conf_threshold,
            min_obs=policy_b_min_obs,
            temporal_window_frames=policy_b_window,
            verifier_threshold=verifier_threshold,
            policy_c_requires_policy_b=False,
        )
        apply_policies(events, pcfg)

        # Persist
        st.write("5. Saving findings and evidence images...")
        run_config["tracker"] = events[0].tracker_config if events else grouper._tracker_config_snapshot()
        run_config["status"] = "complete"
        transfer = db.replace_completed_run(video_id, meta, events, run_config,
                                            actual_frames[0], preserve_reviews)
        if preserve_reviews:
            st.write(f"Matched {transfer['transferred']} previous reviews to new findings. {transfer['unmatched']} could not be matched; earlier results remain in the archive.")
        elapsed_total = time.time() - t0
        status.update(label=f"Completed in {elapsed_total:.1f}s! ({len(events)} events generated)", state="complete", expanded=False)

    st.session_state["analysis_notice"] = f"Analysis complete. {len(events)} possible pothole findings are ready to review."
    st.session_state["pending_rev_filter_vid"] = video_id
    st.session_state["pending_rev_filter_status"] = "Unreviewed"
    st.session_state["pending_nav_page"] = "Review Events"
    st.rerun()


def page_review_events():
    st.subheader("Review findings")
    st.caption("Check the images and video, then save your decision. Only findings marked 'Pothole' appear in the confirmed report.")
    db = get_db()
    settings = current_settings()
    videos = {v["video_id"]: v for v in db.get_videos()}
    notice = st.session_state.pop("review_saved_msg", None)
    if notice:
        st.success(notice)
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        selected_video = st.selectbox("Video", ["All videos"] + list(videos), key="rev_filter_vid",
                                     format_func=lambda vid: video_label(videos[vid]) if vid in videos else vid)
    with c2:
        status = st.selectbox("Review status", ["Unreviewed", "All", "Confirmed", "Rejected", "Uncertain", "Flag Grouping"],
            format_func=lambda value: {"All": "All decisions", "Unreviewed": "Needs review", "Confirmed": "Pothole", "Rejected": "Not a pothole", "Uncertain": "Unsure", "Flag Grouping": "Possible duplicate or grouping issue"}[value], key="rev_filter_status")
    with c3:
        search = st.text_input("Search findings", placeholder="Road name, video, or finding ID", key="rev_search_id").strip().casefold()
    events = db.get_events(video_id=None if selected_video == "All videos" else selected_video, include_reviews=True)
    if not events:
        st.info("No findings are available for this selection. Analyze a video with 'Find potholes to review' to create a review list.")
        if st.button("Analyze a video", type="primary"):
            go_to("Process Video")
        return
    selected_policy = "All candidates"
    with st.expander("More filters"):
        segments = sorted({road_label(e.get("road_segment_id")) for e in events})
        road = st.selectbox("Road or area", ["All areas"] + segments, key="rev_segment")
        order = st.selectbox("Order", ["Time in video", "Latest first"], key="rev_order")
        if settings["research_tools"]:
            selected_policy = st.selectbox("Automated filter", ["All candidates", "Policy A retained", "Policy B retained", "Policy C retained", "Filtered by Policy A", "Filtered by Policy B", "Filtered by Policy C", "Policy C unavailable"], key="rev_filter_policy")

    filtered = []
    for event in events:
        decision = (event.get("review") or {}).get("decision", "unreviewed")
        if status != "All" and decision != status.lower().replace(" ", "_"):
            continue
        if road != "All areas" and road_label(event.get("road_segment_id")) != road:
            continue
        video = videos.get(event["video_id"], {"video_id": event["video_id"]})
        haystack = f"{event['event_id']} {video_label(video)} {event.get('road_segment_id', '')}".casefold()
        if search and search not in haystack:
            continue
        if selected_policy != "All candidates":
            field = "policy_" + selected_policy.split("Policy ")[1][0].lower() + "_retained"
            value = event.get(field)
            if selected_policy == "Policy C unavailable":
                if value is not None:
                    continue
            elif selected_policy.startswith("Filtered"):
                if value != 0:
                    continue
            elif not value:
                continue
        filtered.append(event)
    filtered.sort(key=lambda e: (e["video_id"], e.get("first_timestamp_ms") or 0), reverse=order == "Latest first")
    if not filtered:
        st.info("No findings match these filters. You may have finished this review list.")
        if st.button("Show all findings"):
            for key in ("rev_search_id", "rev_segment", "rev_filter_policy", "rev_order"):
                st.session_state.pop(key, None)
            st.session_state["pending_rev_filter_status"] = "All"
            st.rerun()
        return
    ids = [e["event_id"] for e in filtered]
    if st.session_state.get("selected_event_id") not in ids:
        st.session_state["selected_event_id"] = ids[0]
    event_map = {e["event_id"]: e for e in filtered}
    positions = {eid: index + 1 for index, eid in enumerate(ids)}
    st.caption(f"{len(filtered)} findings in this list")
    def finding_label(eid):
        event = event_map[eid]
        label = DECISION_LABELS.get((event.get("review") or {}).get("decision", "unreviewed"), "Needs review")
        return f"Finding {positions[eid]} | {fmt_ts(event['first_timestamp_ms'])} | {video_label(videos.get(event['video_id'], {'video_id': event['video_id']}))} | {label}"
    st.selectbox("Choose a finding", ids, format_func=finding_label, key="selected_event_id")
    _render_event_detail_panel(event_map[st.session_state["selected_event_id"]], db, filtered)


@st.fragment
def _render_source_video(filepath, timestamp_ms, event_id):
    if st.checkbox("Play video at this finding", key=f"source_video_{event_id}"):
        start = max(0, timestamp_ms / 1000 - 2)
        st.video(filepath, start_time=start)
        st.caption(f"Starts two seconds before the finding at {fmt_ts_full(timestamp_ms)}.")


def _render_event_detail_panel(ev: dict, db: Database, candidate_list: list[dict]):
    eid = ev["event_id"]
    if st.session_state.pop("pending_clear_event", None) == eid:
        for prefix in ("widget_dec_", "widget_fp_", "widget_notes_"):
            st.session_state.pop(prefix + eid, None)
    review = ev.get("review") or {}
    index = [e["event_id"] for e in candidate_list].index(eid)
    heading, previous, following = st.columns([5, 1, 1])
    with heading:
        st.markdown(f"### Finding {index + 1} of {len(candidate_list)}")
        st.caption(f"Time in video: {fmt_ts_full(ev.get('rep_timestamp_ms') or ev['first_timestamp_ms'])} | {DECISION_LABELS.get(review.get('decision', 'unreviewed'), 'Needs review')}")
    with previous:
        if st.button("Previous", disabled=index == 0, width="stretch"):
            st.session_state["pending_selected_event_id"] = candidate_list[index - 1]["event_id"]
            st.rerun()
    with following:
        if st.button("Next", disabled=index == len(candidate_list) - 1, width="stretch"):
            st.session_state["pending_selected_event_id"] = candidate_list[index + 1]["event_id"]
            st.rerun()
    closeup, context = st.columns([1, 2])
    for col, label, path in ((closeup, "Close-up", ev.get("crop_path")), (context, "Road view", ev.get("full_frame_path"))):
        with col:
            st.caption(label)
            if path and Path(path).is_file():
                st.image(path, width="stretch")
            else:
                st.info(f"{label} is unavailable. Use the source video to check this finding.")
    video = db.get_video(ev["video_id"])
    if video and video.get("filepath") and Path(video["filepath"]).is_file():
        _render_source_video(video["filepath"], ev.get("rep_timestamp_ms") or ev["first_timestamp_ms"], eid)
    st.markdown("#### What does this show?")
    st.caption("Choose Unsure if the image is unclear. Flag a possible duplicate if this appears to be the same pothole as another finding.")
    with st.form(f"review_{eid}"):
        options = [None, "confirmed", "rejected", "uncertain", "flag_grouping"]
        current = review.get("decision")
        decision = st.selectbox("Your decision", options, index=options.index(current) if current in options else 0,
            format_func=lambda value: DECISION_LABELS[value] if value else "Choose a decision", key=f"widget_dec_{eid}")
        reason = st.selectbox("Reason if this is not a pothole (optional)", list(REASON_LABELS),
            index=list(REASON_LABELS).index(review.get("fp_category") or "") if (review.get("fp_category") or "") in REASON_LABELS else 0,
            format_func=lambda value: REASON_LABELS[value], key=f"widget_fp_{eid}")
        notes = st.text_area("Notes (optional)", value=review.get("reviewer_notes") or "", key=f"widget_notes_{eid}",
            placeholder="For example: image unclear, or possible duplicate of another finding.")
        save_col, next_col = st.columns(2)
        with save_col:
            save = st.form_submit_button("Save decision", width="stretch")
        with next_col:
            advance = st.form_submit_button("Save and review next", type="primary", width="stretch")
    if save or advance:
        if decision is None:
            st.warning("Choose a decision before saving.")
        else:
            db.save_review(eid, decision, (reason or None) if decision == "rejected" else None, notes)
            message = "Decision saved."
            if advance:
                # Respect the current video, road, search, and automated filters.
                remaining = [e for e in candidate_list[index + 1:] + candidate_list[:index]
                             if e["event_id"] != eid and not e.get("review")]
                if remaining:
                    st.session_state["pending_selected_event_id"] = remaining[0]["event_id"]
                else:
                    message += " No more unreviewed findings in this list."
            st.session_state["review_saved_msg"] = message
            st.rerun()
    with st.expander("Finding reference and saved decision"):
        st.write(f"Finding ID: {eid}")
        st.write(f"Video: {video_label(video) if video else ev['video_id']}")
        st.write(f"Road or area: {road_label(ev.get('road_segment_id'))}")
        if review:
            st.write(f"Saved decision: {DECISION_LABELS.get(review['decision'], review['decision'])}")
            st.write(f"Saved on: {review.get('reviewed_at') or 'Not available'}")
            if st.button("Undo saved decision", key=f"undo_{eid}"):
                db.delete_review(eid)
                st.session_state["pending_clear_event"] = eid
                st.session_state["review_saved_msg"] = "Decision removed. This finding needs review again."
                st.rerun()
    if current_settings()["research_tools"]:
        with st.expander("Technical analysis details"):
            st.caption("Model scores are not probabilities. Automated filters do not change your review decision.")
            st.json({key: ev.get(key) for key in ("observation_count", "detector_conf_max", "detector_conf_mean", "policy_a_retained", "policy_b_retained", "policy_c_score", "policy_c_retained")})


def page_evaluation():
    st.subheader("Research Evaluation")
    st.caption("Event-level benchmark metrics evaluated against independent ground-truth physical defects.")
    db = get_db()

    st.markdown("""
    <div class="info-callout">
      <strong>Research Ground Truth vs. Human Review Decisions:</strong>
      Human reviews on candidate alerts verify whether the detector's proposed reports are valid.
      However, they <em>cannot</em> measure recall or false negatives across unproposed road defects.
      To calculate true event recall, report precision, and false reports per video minute, candidate reports
      must be matched against independently annotated physical defect ground truth (<code>gt_event_id</code>).
    </div>
    """, unsafe_allow_html=True)

    videos = db.get_videos()
    if not videos:
        st.info("No videos registered. Go to **Analyze a video** to analyze a video first.")
        return

    col_v, col_src = st.columns([2, 2])
    with col_v:
        sel_eval_vid = st.selectbox("Select video to evaluate", [v["video_id"] for v in videos], key="eval_sel_vid")
    with col_src:
        gt_source = st.radio("Ground Truth Source", ["Local video annotations (data/annotations/video/)", "Upload ground truth CSV"], horizontal=True)

    gt_annotations: list[GroundTruthAnnotation] = []

    if gt_source.startswith("Local"):
        try:
            gt_annotations = load_video_ground_truth(sel_eval_vid, project_root=PROJECT_ROOT)
        except ValueError as exc:
            st.error(str(exc), icon=":material/error:")
            return
        if gt_annotations:
            num_potholes = len(set(g.gt_event_id for g in gt_annotations if g.is_pothole))
            num_boxes = len([g for g in gt_annotations if g.is_pothole])
            st.success(f"Loaded **{len(gt_annotations)} ground-truth annotations** for `{sel_eval_vid}`: **{num_potholes} verified physical potholes** ({num_boxes} annotated boxes).")
        else:
            st.warning(f"No local ground-truth JSON files found matching `{sel_eval_vid}` in `data/annotations/video/`.")
    else:
        uploaded_csv = st.file_uploader("Upload Ground Truth CSV", type=["csv"], key="eval_gt_csv")
        if uploaded_csv:
            csv_content = uploaded_csv.read().decode("utf-8")
            try:
                gt_annotations = parse_ground_truth_csv(csv_content, default_video_id=sel_eval_vid)
            except ValueError as exc:
                st.error(str(exc), icon=":material/error:")
                return
            num_potholes = len(set(g.gt_event_id for g in gt_annotations if g.is_pothole))
            st.success(f"Parsed {len(gt_annotations)} ground-truth records ({num_potholes} unique physical potholes).")

    # If no ground truth, show honest requirement card
    if not gt_annotations:
        st.markdown("""
        <div class="info-callout warning">
          <strong>Independent Ground Truth Missing:</strong><br/>
          To run event-level evaluation (calculating event recall, report precision, false reports per minute,
          and duplicate counts), independent ground truth annotations containing physical pothole IDs
          (<code>gt_event_id</code>) and timestamps are required.<br/><br/>
          <strong>How to generate ground truth:</strong>
          <ol>
            <li>Use the annotation tool: <code>streamlit run dashboard/annotate.py</code> to label video frames.</li>
            <li>Or upload a ground-truth CSV with columns: <code>timestamp_ms, x1, y1, x2, y2, gt_event_id, label</code>.</li>
          </ol>
          <em>Note: We do not fabricate evaluation metrics or infer recall from reviewed candidate alerts alone.</em>
        </div>
        """, unsafe_allow_html=True)
        return

    # Evaluation settings
    st.markdown('<div class="section-hdr">Evaluation Configuration</div>', unsafe_allow_html=True)
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        eval_iou = st.slider("Spatial IoU Threshold (Matching)", 0.20, 0.80, 0.50, 0.05, help="Primary spatial overlap threshold (standard 0.50).")
    with ec2:
        temporal_tol = st.slider("Temporal Tolerance Δt (ms)", 0, 3000, 0, 50, help="Default: exact timestamps. Increase only when justified by annotation cadence.")
    with ec3:
        vmeta = db.get_video(sel_eval_vid) or {}
        duration = vmeta.get("duration_sec")
        if not duration or duration <= 0:
            st.error("Verified positive video duration is required. Reprocess the source video.")
            return
        eval_dur_min = duration / 60.0
        st.metric("Source duration", f"{eval_dur_min:.3f} min")

    justification = st.text_input("Alignment tolerance justification", placeholder="Required when using a nonzero tolerance") if temporal_tol else "Exact timestamps"
    if temporal_tol and not justification.strip():
        st.info("Document why timestamp tolerance is required before scoring.")
        return
    # ── Run Evaluation on Policy A, Policy B, and Policy C ──
    split = st.selectbox("Evaluation split", ["Unspecified", "Validation", "Held-out test"])
    verified = st.checkbox("I verified physical pothole IDs, complete video coverage, and annotation/video alignment.")
    frozen = st.checkbox("Thresholds were frozen before inspecting held-out test results.") if split == "Held-out test" else True
    st.caption("CSV IDs and timestamps are user-supplied ground truth. Legacy training annotations are not automatically eligible for scoring.")
    if not verified or split == "Unspecified" or not frozen:
        st.info("Confirm ground-truth coverage and select the evaluation split to calculate metrics.")
        return
    all_events = [db.get_event(e["event_id"]) for e in db.get_events(video_id=sel_eval_vid, include_reviews=False)]
    events_a = [e for e in all_events if e.get("policy_a_retained")]
    events_b = [e for e in all_events if e.get("policy_b_retained")]
    events_c = [e for e in all_events if e.get("policy_c_retained")]

    res_a = evaluate_event_detections(events_a, gt_annotations, video_duration_min=eval_dur_min, iou_threshold=eval_iou, temporal_tolerance_ms=temporal_tol)
    res_b = evaluate_event_detections(events_b, gt_annotations, video_duration_min=eval_dur_min, iou_threshold=eval_iou, temporal_tolerance_ms=temporal_tol)

    scored = sum(e.get("policy_c_score") is not None for e in all_events)
    has_c = bool(all_events) and scored == len(all_events)
    st.caption(f"Visual verifier coverage: {scored}/{len(all_events)} candidates scored; {len(all_events)-scored} unavailable.")
    if not has_c:
        st.warning("Policy C comparison unavailable: complete verifier coverage is required.")
    res_c = evaluate_event_detections(events_c, gt_annotations, video_duration_min=eval_dur_min, iou_threshold=eval_iou, temporal_tolerance_ms=temporal_tol) if has_c else None

    st.markdown('<div class="section-hdr">Comparative Policy Evaluation (A vs. B vs. C)</div>', unsafe_allow_html=True)

    comp_data = [
        {
            "Metric": "Total Candidate Reports",
            "Policy A (Detector Only)": res_a.total_reports,
            "Policy B (Temporal Window)": res_b.total_reports,
            "Policy C (Visual Verifier)": res_c.total_reports if res_c else "Unavailable",
        },
        {
            "Metric": "Ground Truth Potholes (Defects)",
            "Policy A (Detector Only)": res_a.total_gt_potholes,
            "Policy B (Temporal Window)": res_b.total_gt_potholes,
            "Policy C (Visual Verifier)": res_c.total_gt_potholes if res_c else "Unavailable",
        },
        {
            "Metric": "True Positives (Unique Defect Matches)",
            "Policy A (Detector Only)": res_a.true_positives,
            "Policy B (Temporal Window)": res_b.true_positives,
            "Policy C (Visual Verifier)": res_c.true_positives if res_c else "Unavailable",
        },
        {
            "Metric": "False Reports (FP)",
            "Policy A (Detector Only)": res_a.false_positives,
            "Policy B (Temporal Window)": res_b.false_positives,
            "Policy C (Visual Verifier)": res_c.false_positives if res_c else "Unavailable",
        },
        {
            "Metric": "Duplicate Retained Reports",
            "Policy A (Detector Only)": res_a.duplicate_reports,
            "Policy B (Temporal Window)": res_b.duplicate_reports,
            "Policy C (Visual Verifier)": res_c.duplicate_reports if res_c else "Unavailable",
        },
        {
            "Metric": "Missed Potholes (FN)",
            "Policy A (Detector Only)": res_a.missed_potholes,
            "Policy B (Temporal Window)": res_b.missed_potholes,
            "Policy C (Visual Verifier)": res_c.missed_potholes if res_c else "Unavailable",
        },
        {
            "Metric": "Event Recall (TP / GT)",
            "Policy A (Detector Only)": f"{res_a.event_recall * 100:.1f}%",
            "Policy B (Temporal Window)": f"{res_b.event_recall * 100:.1f}%",
            "Policy C (Visual Verifier)": f"{res_c.event_recall * 100:.1f}%" if res_c else "Unavailable",
        },
        {
            "Metric": "Report Precision (TP / Total Retained)",
            "Policy A (Detector Only)": f"{res_a.report_precision * 100:.1f}%",
            "Policy B (Temporal Window)": f"{res_b.report_precision * 100:.1f}%",
            "Policy C (Visual Verifier)": f"{res_c.report_precision * 100:.1f}%" if res_c else "Unavailable",
        },
        {
            "Metric": "False Reports / Video Minute",
            "Policy A (Detector Only)": f"{res_a.false_reports_per_minute:.2f} / min",
            "Policy B (Temporal Window)": f"{res_b.false_reports_per_minute:.2f} / min",
            "Policy C (Visual Verifier)": f"{res_c.false_reports_per_minute:.2f} / min" if res_c else "Unavailable",
        },
    ]

    st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

    # ── Sensitivity Analysis (IoU 0.5 vs IoU 0.3) ──
    st.markdown('<div class="section-hdr">IoU Sensitivity Analysis (Threshold 0.5 vs. 0.3)</div>', unsafe_allow_html=True)
    sens_b = run_sensitivity_analysis(events_b, gt_annotations, video_duration_min=eval_dur_min, thresholds=(0.5, 0.3), temporal_tolerance_ms=temporal_tol)

    sens_data = []
    for thresh, r in sens_b.items():
        sens_data.append({
            "IoU Threshold": f"IoU ≥ {thresh:.1f}",
            "True Positives": r.true_positives,
            "False Reports": r.false_positives + r.duplicate_reports,
            "Missed Potholes": r.missed_potholes,
            "Event Recall": f"{r.event_recall * 100:.1f}%",
            "Report Precision": f"{r.report_precision * 100:.1f}%",
            "False Reports / Min": f"{r.false_reports_per_minute:.2f}",
        })
    st.dataframe(pd.DataFrame(sens_data), width="stretch", hide_index=True)

    # ── Look-Alike Breakdown ──
    st.markdown('<div class="section-hdr">Look-Alike False Positive Breakdown (Policy B)</div>', unsafe_allow_html=True)
    lk = res_b.lookalike_breakdown
    col_lk1, col_lk2, col_lk3, col_lk4, col_lk5 = st.columns(5)
    col_lk1.metric("Shadows", lk.get("shadow", 0))
    col_lk2.metric("Repaired Patches", lk.get("repaired_patch", 0))
    col_lk3.metric("Manhole Covers", lk.get("manhole", 0))
    col_lk4.metric("Ordinary Road", lk.get("ordinary_road", 0))
    col_lk5.metric("Unclassified Clear Road", lk.get("unclassified", 0))


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def page_export():
    render_reports(get_db(), research_tools=current_settings()["research_tools"])


def page_settings():
    st.subheader("Settings")
    st.caption("Everyday users can keep the saved defaults. Technical setup is usually handled by the person maintaining the app.")
    settings = current_settings()
    saved_message = st.session_state.pop("settings_saved", None)
    if saved_message:
        st.success(saved_message)
    st.markdown("### Workspace")
    st.caption("Saved settings apply to this installation, including other people using it. Settings are not user permissions.")
    with st.form("workspace_settings"):
        research = st.checkbox("Show research tools", value=settings["research_tools"],
            help="Adds accuracy evaluation, model details, and technical exports. These are not needed for routine road reviews.")
        with st.expander("Technical setup and analysis settings"):
            st.caption("Changes apply to future analyses. Existing videos and findings keep their original results.")
            detector = st.text_input("Pothole model file", value=settings["detector_path"], help="Path to a trained .pt detector whose first class is pothole. Relative paths start at the project folder.")
            verifier = st.text_input("Additional image-check model (optional)", value=settings["verifier_path"], help="Leave blank if no trained verifier is available.")
            c1, c2 = st.columns(2)
            with c1:
                conf = st.slider("Detection threshold", .1, .9, float(settings["detector_conf"]), .05, help="Lower values flag more possible potholes and can include more false alarms. This is a model score, not a probability.")
                fps = st.slider("Video images checked per second", 1.0, 15.0, float(settings["sample_fps"]), .5, help="Used for the review list. Higher values take longer; original video playback speed is unchanged.")
                size = st.selectbox("Analysis image size (pixels)", [416, 640, 960, 1280], index=[416, 640, 960, 1280].index(settings["imgsz"]), help="Larger images require more processing time and memory.")
                tracking = st.checkbox("Keep track labels in marked videos", value=settings["tracking"])
                clips = st.checkbox("Save short evidence clips", value=settings["generate_clips"])
            with c2:
                iou = st.slider("Overlapping box threshold (NMS IoU)", .1, .9, float(settings["detector_iou"]), .05)
                hits = st.number_input("Repeated sightings required", 1, 10, settings["min_obs"])
                window = st.number_input("Number of checked images to compare", 2, 10, settings["temporal_window"])
                vthreshold = st.slider("Additional image-check threshold", .1, .9, float(settings["verifier_threshold"]), .05)
        if st.form_submit_button("Save settings", type="primary"):
            values = dict(detector_path=detector.strip(), verifier_path=verifier.strip(), detector_conf=conf,
                detector_iou=iou, sample_fps=fps, imgsz=size, min_obs=hits, temporal_window=window,
                verifier_threshold=vthreshold, tracking=tracking, generate_clips=clips, research_tools=research)
            try:
                values = validate_settings(values)
                for field in ("detector_path", "verifier_path"):
                    if values[field] and values[field] != settings[field]:
                        path = Path(values[field])
                        path = path if path.is_absolute() else PROJECT_ROOT / path
                        if not path.is_file():
                            raise ValueError(f"The {('pothole model' if field == 'detector_path' else 'additional image-check model')} file does not exist.")
                if values["detector_path"] != settings["detector_path"]:
                    _, setup_error = configured_detector(values)
                    if setup_error:
                        raise ValueError(setup_error)
                save_settings(SETTINGS_PATH, values)
            except (ValueError, OSError) as exc:
                st.error(f"Settings were not saved. {exc}")
            else:
                st.session_state["settings_saved"] = "Settings saved for this installation."
                st.rerun()

    with st.expander("Help with the workflow"):
        st.markdown("1. **Analyze a video** to find possible potholes. Choose a road or area name for the report.\n2. **Review findings** using the close-up, road view, and video. Mark each as Pothole, Not a pothole, or Unsure.\n3. **Reports** downloads confirmed findings by default. You can include other decisions when needed.\n4. **Watch videos** plays recordings created with marked potholes. These do not create a review list.")
    with st.expander("Manage saved findings"):
        st.caption("Remove the current review list and decisions for one video. Evidence files, source videos, marked videos, and archived results are kept.")
        videos = get_db().get_videos()
        if videos:
            by_id = {v["video_id"]: v for v in videos}
            selected = st.selectbox("Video to manage", list(by_id), format_func=lambda vid: video_label(by_id[vid]), key="manage_video")
            confirm = st.checkbox("Permanently delete this video's current findings and review decisions", key=f"delete_confirm_{selected}")
            if st.button("Delete findings and reviews", disabled=not confirm, icon=":material/delete:"):
                get_db().delete_video_data(selected, clear_reviews=True, reset_status=True)
                st.session_state["settings_saved"] = "The selected video's findings and reviews were deleted."
                st.rerun()
        else:
            st.info("No saved findings to manage.")
    if settings["research_tools"]:
        with st.expander("Ground truth annotation tool"):
            st.caption("For independent research labels, launch the separate annotation workspace:")
            st.code("streamlit run dashboard/annotate.py", language="bash")


PAGES = {
    "Overview": page_overview, "Process Video": page_process,
    "Video Library": page_video_library, "Review Events": page_review_events,
    "Export": page_export, "Settings": page_settings, "Evaluation": page_evaluation,
}
NAV_LABELS = {"Overview": "Overview", "Process Video": "Analyze a video", "Video Library": "Watch videos",
              "Review Events": "Review findings", "Export": "Reports", "Settings": "Settings", "Evaluation": "Research evaluation"}


def main():
    for key in ("nav_page", "selected_event_id", "rev_filter_vid", "rev_filter_status"):
        pending = st.session_state.pop(f"pending_{key}", None)
        if pending is not None:
            st.session_state[key] = pending
    settings = current_settings()
    pages = [key for key in PAGES if key != "Evaluation" or settings["research_tools"]]
    if st.session_state.get("nav_page") not in pages:
        st.session_state["nav_page"] = "Overview"
    with st.sidebar:
        st.markdown("### RoadLens")
        st.caption("Road inspection workspace")
        st.divider()
        st.radio("Navigation", options=pages, key="nav_page", label_visibility="collapsed",
                 format_func=lambda page: NAV_LABELS[page])
        st.divider()
        with st.expander("How it works"):
            st.write("Upload a road video, check the possible potholes, and download your team's findings.")
            st.caption("The software suggests areas to check. Your review decides what goes into the confirmed report.")
    notice = st.session_state.pop("analysis_notice", None)
    if notice:
        st.success(notice)
    PAGES[st.session_state["nav_page"]]()


if __name__ == "__main__":
    main()
