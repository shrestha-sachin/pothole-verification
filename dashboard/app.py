"""Pothole Verification System — Main Streamlit Dashboard.

Launch from project root:
    streamlit run dashboard/app.py

Pages:
  1. Overview     — summary statistics and charts
  2. Process      — upload/select video, run YOLO detection, track progress
  3. Events       — review grouped candidate events
  4. Playback     — video playback with detection timeline
  5. Dataset      — video registration and dataset management
  6. Export       — download CSV report
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import time
import uuid as _uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Pothole Verification System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — refined research-grade executive aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── KPI Metric Card ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 20px 20px 16px;
    text-align: center;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 5px;
}
.kpi-value {
    font-size: 1.95rem;
    font-weight: 700;
    line-height: 1.1;
    font-feature-settings: "tnum";
}
.kpi-value.blue   { color: #1e40af; }
.kpi-value.green  { color: #166534; }
.kpi-value.amber  { color: #92400e; }
.kpi-value.red    { color: #991b1b; }
.kpi-value.slate  { color: #334155; }

/* ── Section Divider ── */
.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin: 26px 0 16px;
}

/* ── Status Badges ── */
.status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 9999px;
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    vertical-align: middle;
    border: 1px solid transparent;
}
.badge-confirmed  { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.badge-rejected   { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.badge-uncertain  { background: #fffbeb; color: #854d0e; border-color: #fde68a; }
.badge-unreviewed { background: #f8fafc; color: #475569; border-color: #e2e8f0; }
.badge-policy-pass { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
.badge-policy-fail { background: #f8fafc; color: #64748b; border-color: #e2e8f0; }

/* ── Technical Callout ── */
.callout {
    background: #f8fafc;
    border-left: 3px solid #2563eb;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px;
    color: #1e3a8a;
    font-size: 0.86rem;
    margin: 12px 0;
}
.callout.success { background: #f0fdf4; border-color: #16a34a; color: #166534; }
.callout.warning { background: #fffbeb; border-color: #d97706; color: #92400e; }
.callout.error   { background: #fef2f2; border-color: #dc2626; color: #991b1b; }

/* ── Step Sequence Header ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 20px 0 10px;
}
.step-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: #1e40af;
    color: #ffffff;
    font-weight: 600;
    font-size: 0.78rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-label {
    font-size: 0.92rem;
    font-weight: 600;
    color: #0f172a;
}

/* ── Action Buttons ── */
.stButton > button[kind="primary"],
.stDownloadButton > button {
    border-radius: 6px;
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    padding-top: 1rem;
}

/* ── Clean Table Header ── */
.stDataFrame th {
    background: #f8fafc !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"Missing dependency: {e}. Run: `pip install opencv-python numpy pandas`")
    st.stop()

try:
    from src.database import Database
    from src.detector.video_sampler import iter_frames, get_video_metadata
    from src.tracking.event_grouper import EventGrouper, EvidenceExtractor, CandidateEvent, ObservedDetection
    from src.evaluation.policies import PolicyConfig, apply_policies, summarize_policies
    from src.verifier.interface import load_verifier
except ImportError as e:
    st.error(f"Project import failed: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Paths & globals
# ---------------------------------------------------------------------------
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "evidence"
DB_PATH = PROJECT_ROOT / "results" / "pothole_verification.db"
LOCAL_VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "local_video"
MODELS_DIR = PROJECT_ROOT / "models"

EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_db() -> Database:
    if "db" not in st.session_state:
        st.session_state["db"] = Database(DB_PATH)
    return st.session_state["db"]


def fmt_ts(ms: int) -> str:
    """Format milliseconds as M:SS."""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def fmt_ts_full(ms: int) -> str:
    """Format milliseconds as MM:SS.mmm"""
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}.{ms % 1000:03d}"


def kpi_card(label: str, value, color: str = "blue") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {color}">{value}</div>'
        f'</div>'
    )


def step_header(num: int, label: str):
    st.markdown(
        f'<div class="step-row">'
        f'<div class="step-num">{num}</div>'
        f'<div class="step-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


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


def decision_badge(decision: str) -> str:
    d = decision.lower() if decision else "unreviewed"
    cls_map = {
        "confirmed": "badge-confirmed",
        "rejected": "badge-rejected",
        "uncertain": "badge-uncertain",
        "flag_grouping": "badge-uncertain",
    }
    cls = cls_map.get(d, "badge-unreviewed")
    return f'<span class="status-badge {cls}">{d.upper()}</span>'


def policy_badge(passed: bool | None, label: str) -> str:
    if passed is None:
        return f'<span class="status-badge badge-unreviewed">{label}: N/A</span>'
    cls = "badge-policy-pass" if passed else "badge-policy-fail"
    status_text = "PASS" if passed else "FILTER"
    return f'<span class="status-badge {cls}">{label}: {status_text}</span>'


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
def page_overview():
    st.title("Dashboard Overview")
    db = get_db()
    stats = db.get_summary_stats()

    # ── KPI row 1 ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Videos Processed", stats["total_videos"], "blue"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Candidate Events", stats["total_events"], "slate"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Total Detections", stats["total_detections"], "slate"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Human Reviewed", stats["reviewed"], "amber"), unsafe_allow_html=True)

    st.write("")  # spacer

    # ── KPI row 2 ──
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(kpi_card("Policy A Retained", stats["policy_a_retained"], "blue"), unsafe_allow_html=True)
    with c6:
        st.markdown(kpi_card("Policy B Retained", stats["policy_b_retained"], "blue"), unsafe_allow_html=True)
    with c7:
        st.markdown(kpi_card("Confirmed", stats["confirmed"], "green"), unsafe_allow_html=True)
    with c8:
        st.markdown(kpi_card("Rejected", stats["rejected"], "red"), unsafe_allow_html=True)

    st.write("")

    # ── Charts ──
    events = db.get_events(include_reviews=True)
    if events:
        df = pd.DataFrame(events)
        df["start_sec"] = df["first_timestamp_ms"] / 1000.0
        df["status"] = df["review"].apply(lambda r: r["decision"] if r else "unreviewed")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Detections over video time</div>', unsafe_allow_html=True)
            chart_df = df[["start_sec", "detector_conf_max"]].rename(
                columns={"start_sec": "Time (sec)", "detector_conf_max": "Confidence"}
            ).sort_values("Time (sec)")
            st.bar_chart(chart_df, x="Time (sec)", y="Confidence", height=280, color="#2563eb")

        with col_b:
            st.markdown('<div class="section-title">Review status breakdown</div>', unsafe_allow_html=True)
            counts = df["status"].value_counts().reset_index()
            counts.columns = ["Status", "Count"]
            st.bar_chart(counts, x="Status", y="Count", height=280, color="#2563eb")

        # ── Recent events table ──
        st.markdown('<div class="section-title">Recent candidate events</div>', unsafe_allow_html=True)
        table = df.head(12)[[
            "event_id", "video_id", "start_sec", "observation_count",
            "detector_conf_max", "status"
        ]].copy()
        table.columns = ["Event ID", "Video", "Time (s)", "Observations", "Max Conf", "Status"]
        table["Time (s)"] = table["Time (s)"].map(lambda v: f"{v:.1f}")
        table["Max Conf"] = table["Max Conf"].map(lambda v: f"{v:.3f}")
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No events yet. Go to **Process Video** to analyze a video.")

    # ── Registered videos ──
    videos = db.get_videos()
    if videos:
        st.markdown('<div class="section-title">Registered videos</div>', unsafe_allow_html=True)
        vdf = pd.DataFrame(videos)
        cols = [c for c in [
            "video_id", "filename", "processing_status",
            "duration_sec", "total_frames_sampled", "registered_at"
        ] if c in vdf.columns]
        display = vdf[cols].copy()
        if "duration_sec" in display.columns:
            display["duration_sec"] = display["duration_sec"].apply(
                lambda v: f"{v:.1f}s" if pd.notna(v) else "—"
            )
        st.dataframe(display, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: PROCESS VIDEO
# ═══════════════════════════════════════════════════════════════════════════
def page_process():
    st.title("Process Video")
    st.caption("Upload or select a road video, choose a YOLO checkpoint, configure settings, and run detection.")
    db = get_db()

    # ── Step 1: Model ──
    step_header(1, "Select YOLO model checkpoint")

    candidate_models: list[Path] = []
    # Search models/, weights/, and YOLO training runs (runs/)
    for folder, pattern in [
        (PROJECT_ROOT / "models", "**/*.pt"),
        (PROJECT_ROOT / "weights", "**/*.pt"),
        (PROJECT_ROOT / "runs", "**/*.pt"),
        (PROJECT_ROOT, "*.pt"),
    ]:
        if folder.exists():
            candidate_models.extend(folder.glob(pattern))

    # Deduplicate and sort: prioritize best.pt, then newest modified first
    def _model_sort_key(p: Path) -> tuple[int, float]:
        is_best = 0 if "best" in p.name.lower() else 1
        mtime = -p.stat().st_mtime if p.exists() else 0.0
        return (is_best, mtime)

    candidate_models = sorted(set(candidate_models), key=_model_sort_key)

    def _format_model_choice(p_str: str) -> str:
        try:
            rel = Path(p_str).relative_to(PROJECT_ROOT)
            return str(rel)
        except ValueError:
            return Path(p_str).name

    col_m, col_minfo = st.columns([3, 1])
    with col_m:
        if candidate_models:
            model_choice = st.selectbox(
                "Detected checkpoints",
                options=[str(p) for p in candidate_models],
                format_func=_format_model_choice,
                help="Checkpoints found in runs/, weights/, models/, and project root.",
            )
        else:
            model_choice = ""
            st.warning("No `.pt` files found in `runs/`, `weights/`, `models/`, or project root.")
        model_path_input = st.text_input(
            "Or enter path manually",
            value=model_choice,
            placeholder="/path/to/best.pt",
        )
    with col_minfo:
        mp = Path(model_path_input.strip()) if model_path_input.strip() else None
        if mp and not mp.is_absolute() and not mp.exists() and (PROJECT_ROOT / mp).exists():
            mp = PROJECT_ROOT / mp
        model_valid = False
        if mp and mp.exists():
            st.metric("Checkpoint", f"{mp.stat().st_size / (1024*1024):.1f} MB")
            try:
                import torch
                ckpt = torch.load(str(mp), map_location="cpu", weights_only=False)
                names = getattr(ckpt.get("model"), "names", None) if isinstance(ckpt, dict) else None
                if names is None and isinstance(ckpt, dict):
                    names = ckpt.get("names")
                if names and isinstance(names, dict):
                    cls0 = names.get(0, "")
                    if "pothole" in str(cls0).lower():
                        st.success(f"`{mp.name}`: class 0 is '{cls0}' ✓")
                        model_valid = True
                    else:
                        st.error(f"`{mp.name}`: class 0 is '{cls0}' (not 'pothole'). Please select a fine-tuned pothole detector (e.g. models/best.pt).")
                        model_valid = False
                else:
                    st.success(f"`{mp.name}` found")
                    model_valid = True
            except Exception:
                st.success(f"`{mp.name}` found")
                model_valid = True
        elif model_path_input.strip():
            st.error("File not found")

    # ── Step 2: Verifier ──
    step_header(2, "Verifier checkpoint (optional)")
    verifier_models = sorted(MODELS_DIR.glob("verifier*.pt")) if MODELS_DIR.exists() else []
    verifier_path_input = st.text_input(
        "MobileNetV3-Small verifier path (leave blank if not trained yet)",
        value=str(verifier_models[0]) if verifier_models else "",
        placeholder="models/verifier_v1.pt",
    )
    if not verifier_path_input.strip():
        st.info("Verifier not configured — Policy C will be skipped. This is expected until you train MobileNetV3-Small.")

    # ── Step 3: Video ──
    step_header(3, "Select or upload video")

    tab_local, tab_upload = st.tabs(["Local File", "Upload Video"])

    video_path_final: Path | None = None
    video_id_final: str = ""

    with tab_local:
        local_videos = []
        if LOCAL_VIDEO_DIR.exists():
            local_videos = sorted(LOCAL_VIDEO_DIR.glob("*.mp4")) + sorted(LOCAL_VIDEO_DIR.glob("*.mov"))
        if local_videos:
            choice = st.selectbox(
                "Available videos",
                options=[str(p) for p in local_videos],
                format_func=lambda p: Path(p).name,
            )
            override = st.text_input("Or enter path manually", value="", key="local_path_override")
            chosen = override.strip() or choice
        else:
            chosen = st.text_input("Video path", placeholder="data/raw/local_video/your_video.mp4")

        if chosen:
            p = Path(chosen) if Path(chosen).is_absolute() else PROJECT_ROOT / chosen
            if p.exists():
                video_path_final = p
                video_id_final = st.text_input("Video ID", value=p.stem.replace(" ", "_"),
                                               help="Short identifier used in the database.")
                meta = get_video_metadata(str(p))
                if "error" not in meta:
                    mc = st.columns(4)
                    mc[0].metric("Duration", f"{meta['duration_min']:.1f} min")
                    mc[1].metric("Resolution", f"{meta['width']}×{meta['height']}")
                    mc[2].metric("Source FPS", f"{meta['fps']:.1f}")
                    mc[3].metric("File size", f"{meta['file_size_mb']:.0f} MB")

                    with st.expander("Preview first frame"):
                        cap = cv2.VideoCapture(str(p))
                        ok, frame = cap.read()
                        cap.release()
                        if ok:
                            st.image(
                                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                caption=p.name,
                                use_container_width=True,
                            )
            else:
                st.error(f"File not found: `{p}`")

    with tab_upload:
        uploaded = st.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
        if uploaded:
            LOCAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            save_path = LOCAL_VIDEO_DIR / uploaded.name
            with open(save_path, "wb") as fh:
                fh.write(uploaded.read())
            video_path_final = save_path
            video_id_final = save_path.stem.replace(" ", "_")
            st.success(f"Saved to `{save_path.relative_to(PROJECT_ROOT)}`")
            meta = get_video_metadata(str(save_path))
            if "error" not in meta:
                c1, c2 = st.columns(2)
                c1.metric("Duration", f"{meta['duration_min']:.1f} min")
                c2.metric("Resolution", f"{meta['width']}×{meta['height']}")

    # ── Step 4: Settings ──
    step_header(4, "Inference settings")

    with st.container():
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            sample_fps = st.slider("Sample FPS", 1.0, 15.0, 5.0, 0.5,
                                   help="How many frames per second to extract for inference.")
        with s2:
            conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05,
                                       help="Minimum YOLO confidence to keep a detection.")
        with s3:
            iou_threshold = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
        with s4:
            imgsz = st.selectbox("Inference size (px)", [416, 640, 960, 1280], index=1)

        p1, p2, p3 = st.columns(3)
        with p1:
            policy_b_min_obs = st.number_input("Policy B min observations", 1, 10, 2,
                                               help="Minimum detections to pass temporal verification.")
        with p2:
            road_segment_id = st.text_input("Road segment ID", value="seg_001")
        with p3:
            verifier_threshold = st.slider("Verifier threshold (Policy C)", 0.10, 0.90, 0.50, 0.05)

        opt1, opt2 = st.columns(2)
        with opt1:
            preserve_reviews = st.checkbox("Preserve human review decisions on reprocessing", value=True,
                                           help="Retains existing reviews for this video when re-running pipeline.")
        with opt2:
            generate_clips = st.checkbox("Generate short evidence video clips (MP4)", value=False,
                                         help="Extracts 3-second video clips around each candidate event.")

    # ── Run ──
    st.divider()
    can_run = model_valid and video_path_final is not None
    if not model_valid:
        st.warning("Select a valid model checkpoint to continue.")
    if video_path_final is None:
        st.warning("Select or upload a video to continue.")

    if st.button("Execute Pipeline", disabled=not can_run, type="primary", use_container_width=True):
        chosen_mp = Path(model_path_input.strip())
        if not chosen_mp.is_absolute() and not chosen_mp.exists() and (PROJECT_ROOT / chosen_mp).exists():
            chosen_mp = PROJECT_ROOT / chosen_mp
        _run_analysis(
            db, video_path_final, video_id_final, chosen_mp,
            verifier_path_input.strip() or None, sample_fps, conf_threshold,
            iou_threshold, int(imgsz), road_segment_id, int(policy_b_min_obs),
            verifier_threshold, preserve_reviews=preserve_reviews, generate_clips=generate_clips,
        )


def _run_analysis(
    db: Database, video_path: Path, video_id: str, model_path: Path,
    verifier_path: str | None, sample_fps: float, conf_threshold: float,
    iou_threshold: float, imgsz: int, road_segment_id: str,
    policy_b_min_obs: int, verifier_threshold: float,
    preserve_reviews: bool = True, generate_clips: bool = False,
) -> None:
    """Full shared pipeline: sample → detect+track → group → evidence → policies → save."""
    from ultralytics import YOLO

    status_text = st.empty()
    progress = st.progress(0.0)
    detail = st.empty()

    def step(pct: float, msg: str):
        progress.progress(min(1.0, pct))
        status_text.markdown(f"**{msg}**")

    # Register / reset
    meta = get_video_metadata(str(video_path))
    meta["filepath"] = str(video_path)
    meta["road_segment_id"] = road_segment_id
    db.register_video(video_id, meta)
    preserved_reviews = db.delete_events_for_video(video_id, clear_reviews=not preserve_reviews)
    db.mark_processing_started(video_id, {
        "sample_fps": sample_fps, "model_path": str(model_path),
        "detector_conf": conf_threshold, "detector_iou": iou_threshold,
    })

    try:
        t0 = time.time()

        # 1. Load model and validate checkpoint
        step(0.04, "Loading YOLO model weights...")
        model = YOLO(str(model_path))
        device = auto_device()

        grouper = EventGrouper(
            video_id=video_id,
            road_segment_id=road_segment_id,
            policy_b_min_obs=policy_b_min_obs,
            policy_b_window_frames=5,
        )
        grouper.validate_checkpoint(model)
        detail.info(f"Model `{model_path.name}` verified and loaded on {device.upper()}")

        # 2. Process video stream incrementally with bounded memory
        total_sec = meta.get("duration_sec", 0) or 0
        expected = max(1, int(total_sec * sample_fps))
        step(0.12, f"Streaming {total_sec:.1f}s video at {sample_fps} FPS (bounded memory)...")

        def on_progress(fi, total):
            pct = 0.12 + 0.68 * min(1.0, fi / expected)
            step(pct, f"Detecting and tracking frame {fi} of ~{expected}...")

        frame_iter = iter_frames(str(video_path), target_fps=sample_fps)

        events = grouper.process_video_stream(
            frame_iter=frame_iter,
            model=model,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=device,
            evidence_root=EVIDENCE_ROOT,
            video_path=video_path,
            generate_clips=generate_clips,
            total_expected_frames=expected,
            progress_callback=on_progress,
        )

        n_events = len(events)
        detail.info(f"Grouped into **{n_events}** candidate events using BoT-SORT.")

        # 3. Secondary MobileNetV3 verifier (Policy C)
        verifier = load_verifier(verifier_path, threshold=verifier_threshold)
        if verifier.is_configured():
            step(0.85, "Evaluating secondary MobileNetV3 verifier...")
            for ev in events:
                if ev.crop_path and Path(ev.crop_path).exists():
                    cbgr = cv2.imread(ev.crop_path)
                    if cbgr is not None:
                        res = verifier.score_crop(cbgr)
                        ev.policy_c_score = res.pothole_score
                        ev.policy_c_retained = res.is_pothole
        else:
            for ev in events:
                ev.policy_c_score = None
                ev.policy_c_retained = None

        # 4. Apply reporting policies
        step(0.92, "Evaluating reporting suppression policies...")
        pcfg = PolicyConfig(
            conf_threshold=conf_threshold,
            min_obs=policy_b_min_obs,
            temporal_window_frames=5,
            verifier_threshold=verifier_threshold,
            policy_c_requires_policy_b=False,
        )
        apply_policies(events, pcfg)

        # 5. Persist to database and re-attach preserved reviews
        step(0.96, "Persisting events and metadata to SQLite database...")
        db.save_events(events)
        if preserved_reviews:
            reattached = db.reattach_preserved_reviews(video_id, preserved_reviews)
            if reattached:
                detail.info(f"Re-attached {reattached} existing human review decisions.")
        db.mark_processing_done(video_id, expected)

        elapsed = time.time() - t0
        step(1.0, "Analysis complete.")
        progress.empty()
        status_text.empty()

        st.success(
            f"Execution finished in {elapsed:.1f}s. Generated {len(events)} candidate events "
            f"from sampled video frames."
        )

        summary = summarize_policies(events, pcfg)
        r1, r2, r3 = st.columns(3)
        r1.metric("Policy A (detector only)", f"{summary['policy_a_retained']} / {summary['total_candidates']}")
        r2.metric("Policy B (temporal)", f"{summary['policy_b_retained']} / {summary['total_candidates']}")
        r3.metric(
            "Policy C (verifier)",
            f"{summary['policy_c_retained']} / {summary['policy_c_scored']}"
            if verifier.is_configured() else "Not configured",
        )
        st.info("Go to **Review Events** to inspect and label the candidate events.")

    except Exception as exc:
        db.mark_processing_error(video_id, str(exc))
        progress.empty()
        status_text.empty()
        st.error(f"Processing failed: {exc}")
        logger.exception("Processing error")
        raise


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: EVENT REVIEW
# ═══════════════════════════════════════════════════════════════════════════
def page_events():
    st.title("Review Candidate Events")
    db = get_db()

    # ── Sidebar filters ──
    with st.sidebar:
        st.subheader("Filters")
        videos = db.get_videos()
        vid_opts = ["All videos"] + [v["video_id"] for v in videos]
        sel_vid = st.selectbox("Video", vid_opts)

        policy_filter = st.selectbox("Policy filter", [
            "All candidates", "Policy A retained", "Policy B retained",
            "Policy C retained", "Filtered by Policy A", "Filtered by Policy B",
        ])
        review_filter = st.selectbox("Review status", [
            "All", "Unreviewed", "Confirmed", "Rejected", "Uncertain",
        ])
        sort_by = st.selectbox("Sort by", [
            "Time (earliest)", "Confidence (highest)", "Observations (most)",
        ])

    events = db.get_events(
        video_id=None if sel_vid == "All videos" else sel_vid,
        include_reviews=True,
    )

    # Apply filters
    if policy_filter == "Policy A retained":
        events = [e for e in events if e.get("policy_a_retained")]
    elif policy_filter == "Policy B retained":
        events = [e for e in events if e.get("policy_b_retained")]
    elif policy_filter == "Policy C retained":
        events = [e for e in events if e.get("policy_c_retained")]
    elif policy_filter == "Filtered by Policy A":
        events = [e for e in events if not e.get("policy_a_retained")]
    elif policy_filter == "Filtered by Policy B":
        events = [e for e in events if not e.get("policy_b_retained")]

    if review_filter != "All":
        target = review_filter.lower()
        if target == "unreviewed":
            events = [e for e in events if e.get("review") is None]
        else:
            events = [e for e in events if e.get("review") and e["review"]["decision"] == target]

    # Sort
    if sort_by.startswith("Confidence"):
        events.sort(key=lambda e: -(e.get("detector_conf_max") or 0))
    elif sort_by.startswith("Observations"):
        events.sort(key=lambda e: -(e.get("observation_count") or 0))
    else:
        events.sort(key=lambda e: e.get("first_timestamp_ms") or 0)

    st.caption(f"{len(events)} events matching filters")

    if not events:
        st.info("No events found. Process a video first, or adjust the filters.")
        return

    # ── Render each event ──
    for ev in events:
        _render_event(ev, db)


def _render_event(ev: dict, db: Database):
    review = ev.get("review") or {}
    decision = review.get("decision", "unreviewed")
    eid = ev.get("event_id", "???")
    ts = fmt_ts_full(ev.get("first_timestamp_ms", 0))
    conf = ev.get("detector_conf_max", 0) or 0
    nobs = ev.get("observation_count", 0) or 0

    # Build a compact header
    pa = "A:PASS" if ev.get("policy_a_retained") else "A:FILTER"
    pb = "B:PASS" if ev.get("policy_b_retained") else "B:FILTER"
    header = f"Event {eid}  |  {ts}  |  Conf {conf:.3f}  |  {nobs} obs  |  {pa}  {pb}  |  {decision.upper()}"

    with st.expander(header, expanded=False):
        col_img, col_detail, col_review = st.columns([1, 1, 1])

        # ── Image column ──
        with col_img:
            crop_path = ev.get("crop_path", "")
            full_path = ev.get("full_frame_path", "")
            if crop_path and Path(crop_path).exists():
                st.image(str(crop_path), caption="Candidate crop (15% padding)", use_container_width=True)
            elif full_path and Path(full_path).exists():
                st.image(str(full_path), caption="Annotated frame", use_container_width=True)
            else:
                st.caption("No image available")

            if full_path and Path(full_path).exists() and crop_path:
                with st.expander("Show full annotated frame"):
                    st.image(str(full_path), use_container_width=True)

        # ── Detail column ──
        with col_detail:
            st.markdown(decision_badge(decision), unsafe_allow_html=True)
            st.write("")

            st.markdown(f"**Event ID:** `{eid}`")
            st.markdown(f"**Video:** `{ev.get('video_id', '')}`")
            te = fmt_ts_full(ev.get("last_timestamp_ms", 0))
            dur = (ev.get("duration_ms") or 0) / 1000.0
            st.markdown(f"**Time:** {ts} → {te}  ({dur:.2f}s)")
            st.markdown(f"**Observations:** {nobs}")
            st.markdown(f"**Max confidence:** {conf:.4f}")
            mean_c = ev.get("detector_conf_mean", 0) or 0
            st.markdown(f"**Mean confidence:** {mean_c:.4f}")

            # Policy badges
            pa_html = policy_badge(bool(ev.get("policy_a_retained")), "A")
            pb_html = policy_badge(bool(ev.get("policy_b_retained")), "B")
            pc_score = ev.get("policy_c_score")
            if pc_score is not None:
                pc_html = policy_badge(bool(ev.get("policy_c_retained")), f"C ({pc_score:.3f})")
            else:
                pc_html = policy_badge(None, "C")
            st.markdown(f"{pa_html} {pb_html} {pc_html}", unsafe_allow_html=True)

            st.markdown(
                f"**Box:** ({ev.get('rep_x1')}, {ev.get('rep_y1')}) → "
                f"({ev.get('rep_x2')}, {ev.get('rep_y2')})"
            )

        # ── Review column ──
        with col_review:
            st.markdown("**Human Review**")

            options = ["unreviewed", "confirmed", "rejected", "uncertain", "flag_grouping"]
            idx = options.index(decision) if decision in options else 0
            new_dec = st.selectbox("Decision", options, index=idx, key=f"dec_{eid}")

            fp_cat = None
            if new_dec == "rejected":
                fp_opts = ["", "shadow", "repaired_patch", "manhole", "ordinary_road", "other"]
                current_fp = (review.get("fp_category") or "")
                fp_idx = fp_opts.index(current_fp) if current_fp in fp_opts else 0
                fp_cat = st.selectbox("FP category", fp_opts, index=fp_idx, key=f"fp_{eid}")

            notes = st.text_area(
                "Notes", value=review.get("reviewer_notes", "") or "",
                height=70, key=f"notes_{eid}",
            )

            bc1, bc2 = st.columns(2)
            with bc1:
                if st.button("Save Decision", key=f"save_{eid}", use_container_width=True):
                    if new_dec == "unreviewed":
                        db.delete_review(eid)
                    else:
                        db.save_review(eid, new_dec, fp_cat or None, notes)
                    st.rerun()
            with bc2:
                if st.button("Clear Review", key=f"clr_{eid}", use_container_width=True):
                    db.delete_review(eid)
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: VIDEO PLAYBACK
# ═══════════════════════════════════════════════════════════════════════════
def page_video():
    st.title("Video Playback")
    db = get_db()

    videos = db.get_videos()
    if not videos:
        st.info("No videos registered. Go to **Process Video** first.")
        return

    sel_id = st.selectbox("Select video", [v["video_id"] for v in videos])
    vrow = db.get_video(sel_id)
    if not vrow:
        st.error("Video not found.")
        return

    vpath = Path(vrow["filepath"])
    events = db.get_events(video_id=sel_id, include_reviews=True)

    col_v, col_t = st.columns([3, 2])

    with col_v:
        st.subheader("Original video")
        if vpath.exists():
            st.video(str(vpath))
        else:
            st.error(f"Video file not found: `{vpath}`")

    with col_t:
        st.subheader("Detection timeline")
        if events:
            for ev in sorted(events, key=lambda e: e.get("first_timestamp_ms", 0)):
                ts = fmt_ts(ev.get("first_timestamp_ms", 0))
                c = ev.get("detector_conf_max", 0) or 0
                n = ev.get("observation_count", 0) or 0
                rev = (ev.get("review") or {}).get("decision", "—")
                st.markdown(f"**{ts}** · conf `{c:.3f}` · {n} obs · {rev} · `{ev['event_id']}`")
                cp = ev.get("crop_path", "")
                if cp and Path(cp).exists():
                    st.image(str(cp), width=200)
                st.divider()
        else:
            st.info("No events for this video.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: DATASET
# ═══════════════════════════════════════════════════════════════════════════
def page_dataset():
    st.title("Dataset Management")
    db = get_db()

    tab_reg, tab_crops = st.tabs(["Register Video", "Crop Export"])

    with tab_reg:
        st.subheader("Register a road video")
        st.caption(
            "Register videos to track your inventory. This is independent of processing."
        )
        c1, c2 = st.columns(2)
        with c1:
            reg_path = st.text_input("Video filepath", placeholder="data/raw/local_video/seg_003.mp4")
            reg_id = st.text_input("Video ID", placeholder="seg_003_gbay")
            reg_seg = st.text_input("Road segment ID", value="seg_001")
        with c2:
            reg_split = st.selectbox("Dataset split", ["unassigned", "train", "val", "test"])
            st.markdown(
                '<div class="callout warning">'
                'Assign test split <b>before</b> any threshold tuning. '
                'Never use test footage for training, augmentation, or threshold selection.'
                '</div>',
                unsafe_allow_html=True,
            )

        if st.button("Register video", type="primary"):
            if reg_path and reg_id:
                p = Path(reg_path) if Path(reg_path).is_absolute() else PROJECT_ROOT / reg_path
                if p.exists():
                    m = get_video_metadata(str(p))
                    m["filepath"] = str(p)
                    m["road_segment_id"] = reg_seg
                    db.register_video(reg_id, m)
                    st.success(f"Registered `{reg_id}`")
                else:
                    st.error(f"File not found: `{p}`")
            else:
                st.warning("Provide both filepath and video ID.")

    with tab_crops:
        st.subheader("Export verifier crops")
        st.caption("Crops are saved during processing. Use this to export a manifest of reviewed crops.")
        events = db.get_events(include_reviews=True)
        confirmed = [e for e in events if e.get("review") and e["review"]["decision"] == "confirmed"]
        rejected = [e for e in events if e.get("review") and e["review"]["decision"] == "rejected"]

        c1, c2 = st.columns(2)
        c1.metric("Confirmed (positive)", len(confirmed))
        c2.metric("Rejected (negative)", len(rejected))

        if confirmed or rejected:
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=[
                "event_id", "video_id", "label", "fp_category", "crop_path",
                "first_timestamp_ms", "detector_conf_max"
            ])
            w.writeheader()
            for ev in confirmed:
                w.writerow({
                    "event_id": ev["event_id"], "video_id": ev["video_id"],
                    "label": "pothole", "fp_category": "",
                    "crop_path": ev.get("crop_path", ""),
                    "first_timestamp_ms": ev["first_timestamp_ms"],
                    "detector_conf_max": ev["detector_conf_max"],
                })
            for ev in rejected:
                rv = ev.get("review") or {}
                w.writerow({
                    "event_id": ev["event_id"], "video_id": ev["video_id"],
                    "label": "non_pothole", "fp_category": rv.get("fp_category", ""),
                    "crop_path": ev.get("crop_path", ""),
                    "first_timestamp_ms": ev["first_timestamp_ms"],
                    "detector_conf_max": ev["detector_conf_max"],
                })
            st.download_button(
                "Download crop manifest CSV", data=buf.getvalue(),
                file_name="crop_manifest.csv", mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 6: EXPORT
# ═══════════════════════════════════════════════════════════════════════════
def page_export():
    st.title("Export Results")
    db = get_db()

    videos = db.get_videos()
    opts = ["All videos"] + [v["video_id"] for v in videos]
    sel = st.selectbox("Export events for", opts)
    vid_id = None if sel == "All videos" else sel

    events = db.get_events(video_id=vid_id, include_reviews=True)
    st.caption(f"{len(events)} events will be exported.")

    if events:
        csv_str = db.export_events_csv(video_id=vid_id)
        st.download_button(
            "Download Event Report (CSV)", data=csv_str,
            file_name=f"pothole_events_{sel.replace(' ', '_')}.csv",
            mime="text/csv", type="primary", use_container_width=True,
        )
        st.write("")
        df = pd.read_csv(io.StringIO(csv_str))
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No events to export. Process and review a video first.")


# ═══════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
PAGES = {
    "Overview": page_overview,
    "Inference Pipeline": page_process,
    "Event Verification": page_events,
    "Video Playback": page_video,
    "Dataset Management": page_dataset,
    "Export Results": page_export,
}


def main():
    with st.sidebar:
        st.markdown("### Pothole Verification")
        st.caption("Visual Verification & Spatiotemporal Analysis")
        st.divider()

        page = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

        st.divider()
        db = get_db()
        stats = db.get_summary_stats()
        st.caption(
            f"**Database:** `{DB_PATH.name}`\n\n"
            f"**Videos:** {stats['total_videos']}  |  **Events:** {stats['total_events']}  |  "
            f"**Reviewed:** {stats['reviewed']}"
        )

    PAGES[page]()


if __name__ == "__main__":
    main()
