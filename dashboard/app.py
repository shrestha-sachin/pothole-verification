"""Pothole Verification System — Main Streamlit Dashboard.

Launch from project root:
    streamlit run dashboard/app.py

Primary Pages:
  1. Overview      — Executive summary, KPI metrics, review progress, next action
  2. Process Video — Streamlined upload/selection, checkpoint validation, bounded execution
  3. Review Events — Central master/detail candidate review, synced video, policy breakdown
  4. Evaluation    — Rigorous research metrics against independent ground truth (A/B/C comparison)
  5. Export        — Filtered CSV report and crop manifest download
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
import sys
import time
import uuid
import shutil
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root & sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Pothole Verification Research Dashboard",
    page_icon=":material/road:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme-Compatible Research-Grade CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Use system typography: no remote font request or fragile generated selectors. */
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
.kpi-card { border: 1px solid rgba(128,145,165,.25); border-radius: 12px;
    padding: 20px; text-align: left; background: rgba(128,145,165,.035); }
.kpi-label { font-size: .75rem; font-weight: 600; letter-spacing: .04em;
    color: inherit; opacity: .75; margin-bottom: 12px; }
.kpi-value { font-size: 2rem; font-weight: 650; line-height: 1.15; color: inherit;
    font-variant-numeric: tabular-nums; }
.kpi-value.blue { color: #4385e0; }
.kpi-value.green { color: #279d78; }
.kpi-value.amber { color: #bd8c30; }
.kpi-value.red { color: #d46767; }
.kpi-value.slate { color: inherit; }
/* Status Badges */
.status-pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    vertical-align: middle;
    border: 1px solid transparent;
    line-height: 1.2;
}
.pill-confirmed   { background: #dcfce7; color: #15803d; border-color: #86efac; }
.pill-rejected    { background: #fee2e2; color: #b91c1c; border-color: #fca5a5; }
.pill-uncertain   { background: #fef3c7; color: #b45309; border-color: #fde68a; }
.pill-flag        { background: #f1f5f9; color: #475569; border-color: #cbd5e1; }
.pill-unreviewed  { background: #f8fafc; color: #64748b; border-color: #e2e8f0; }
.pill-pass        { background: #dbeafe; color: #1d4ed8; border-color: #bfdbfe; }
.pill-filter      { background: #f1f5f9; color: #64748b; border-color: #cbd5e1; }
.pill-unavailable { background: #fef3c7; color: #92400e; border-color: #fde68a; }

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

/* Master Item Card */
.master-item {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.03);
}
.master-item.selected {
    border-color: #2563eb;
    background: rgba(37, 99, 235, 0.06);
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
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"Missing core dependency: {e}. Run: `pip install opencv-python numpy pandas`")
    st.stop()

try:
    from src.database import Database
    from src.detector.video_sampler import iter_frames, get_video_metadata
    from src.tracking.event_grouper import EventGrouper, CandidateEvent, ObservedDetection
    from src.evaluation.policies import PolicyConfig, apply_policies, summarize_policies
    from src.verifier.interface import load_verifier
    from src.evaluation.matcher import (
        GroundTruthAnnotation,
        evaluate_event_detections,
        run_sensitivity_analysis,
        load_video_ground_truth,
        parse_ground_truth_csv,
    )
    from src.video.annotated_renderer import (
        AnnotatedVideoRenderer,
        RenderMetadata,
        RenderProgress,
        is_ffmpeg_available,
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

EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
ANNOTATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State & Helpers
# ---------------------------------------------------------------------------
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
    return records


def init_session_state():
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Overview"
    if "selected_event_id" not in st.session_state:
        st.session_state["selected_event_id"] = None
    if "review_notes_draft" not in st.session_state:
        st.session_state["review_notes_draft"] = {}
    if "review_saved_msg" not in st.session_state:
        st.session_state["review_saved_msg"] = None


init_session_state()


def get_evidence_disk_usage(video_id: str | None = None) -> tuple[int, float]:
    """Calculate file count and size (MB) of evidence on disk."""
    target_dir = (EVIDENCE_ROOT / video_id) if video_id else EVIDENCE_ROOT
    if not target_dir.exists():
        return 0, 0.0
    file_count = 0
    total_bytes = 0
    for p in target_dir.rglob("*"):
        if p.is_file():
            file_count += 1
            total_bytes += p.stat().st_size
    return file_count, total_bytes / (1024 * 1024)


def delete_evidence_disk(video_id: str | None = None) -> tuple[int, float]:
    """Remove evidence folder for a specific video or all videos. Returns (file_count, size_mb)."""
    import shutil
    target_dir = (EVIDENCE_ROOT / video_id) if video_id else EVIDENCE_ROOT
    if not target_dir.exists():
        return 0, 0.0
    file_count, size_mb = get_evidence_disk_usage(video_id)
    if video_id:
        shutil.rmtree(target_dir, ignore_errors=True)
    else:
        for item in target_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            elif item.is_file():
                item.unlink(missing_ok=True)
    return file_count, size_mb


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


def render_decision_badge(decision: Optional[str]) -> str:
    d = (decision or "unreviewed").lower()
    mapping = {
        "confirmed": ("pill-confirmed", "CONFIRMED"),
        "rejected": ("pill-rejected", "REJECTED"),
        "uncertain": ("pill-uncertain", "UNCERTAIN"),
        "flag_grouping": ("pill-flag", "FLAG GROUPING"),
    }
    css, label = mapping.get(d, ("pill-unreviewed", "UNREVIEWED"))
    return f'<span class="status-pill {css}">{label}</span>'


def render_policy_badge(passed: Optional[bool], code: str, score: Optional[float] = None) -> str:
    if passed is None:
        return f'<span class="status-pill pill-unavailable">{code}: UNAVAILABLE</span>'
    if passed:
        suffix = f" ({score:.2f})" if score is not None else ""
        return f'<span class="status-pill pill-pass">{code}: PASS{suffix}</span>'
    suffix = f" ({score:.2f})" if score is not None else ""
    return f'<span class="status-pill pill-filter">{code}: FILTER{suffix}</span>'


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
def page_overview():
    st.subheader("Overview", anchor=False)
    st.caption("ROAD INTELLIGENCE / RESEARCH WORKSPACE")
    db = get_db()
    stats = db.get_summary_stats()

    # ── KPI Cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Completed Videos</div>'
            f'<div class="kpi-value blue">{stats["total_videos"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Candidate Events</div>'
            f'<div class="kpi-value slate">{stats["total_events"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        rev_pct = (stats["reviewed"] / stats["total_events"] * 100) if stats["total_events"] > 0 else 0.0
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Human Reviewed</div>'
            f'<div class="kpi-value amber">{stats["reviewed"]} <span style="font-size:1rem;color:#64748b">({rev_pct:.0f}%)</span></div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        unrev = stats["total_events"] - stats["reviewed"]
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Unreviewed Events</div>'
            f'<div class="kpi-value slate">{unrev}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Confirmed Potholes</div>'
            f'<div class="kpi-value green">{stats["confirmed"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Rejected False Alarms</div>'
            f'<div class="kpi-value red">{stats["rejected"]}</div></div>',
            unsafe_allow_html=True,
        )
    with c7:
        uncertain_total = stats.get("uncertain", 0) + stats.get("flag_grouping", 0)
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Uncertain / Flagged</div>'
            f'<div class="kpi-value amber">{uncertain_total}</div></div>',
            unsafe_allow_html=True,
        )
    with c8:
        b_pct = (stats["policy_b_retained"] / stats["total_events"] * 100) if stats["total_events"] > 0 else 0.0
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Policy B Retention</div>'
            f'<div class="kpi-value blue">{stats["policy_b_retained"]} <span style="font-size:1rem;color:#64748b">({b_pct:.0f}%)</span></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="info-callout">
      <strong>Model Outcomes vs. Human Decisions:</strong>
      Policies A, B, and C automatically filter raw detector proposals based on confidence, temporal consistency,
      and visual verification. Confirmed, Rejected, and Uncertain ratings are authoritative human review decisions
      recorded to evaluate pipeline precision and train downstream verification classifiers.
    </div>
    """, unsafe_allow_html=True)

    # ── Recommended Next Action Card ──
    st.markdown('<div class="section-hdr">Recommended Next Action</div>', unsafe_allow_html=True)
    unreviewed_count = stats["total_events"] - stats["reviewed"]
    if stats["total_videos"] == 0:
        st.info("No videos processed yet. Go to **Process Video** to analyze a road recording.")
        if st.button("Go to Process Video", icon=":material/upload_file:", type="primary"):
            st.session_state["pending_nav_page"] = "Process Video"
            st.rerun()
    elif unreviewed_count > 0:
        col_act1, col_act2 = st.columns([4, 1])
        with col_act1:
            st.warning(f"There are **{unreviewed_count} unreviewed candidate events** awaiting inspection.")
        with col_act2:
            if st.button("Review Events Now", icon=":material/fact_check:", type="primary", width="stretch"):
                st.session_state["pending_nav_page"] = "Review Events"
                st.rerun()
    else:
        st.success("All proposed candidate events have been reviewed! You can evaluate results or export reports.")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if st.button("View Research Evaluation", width="stretch"):
                st.session_state["pending_nav_page"] = "Evaluation"
                st.rerun()
        with col_e2:
            if st.button("Export Results", width="stretch"):
                st.session_state["pending_nav_page"] = "Export"
                st.rerun()

    # ── Charts Section ──
    events = db.get_events(include_reviews=True)
    if events:
        st.markdown('<div class="section-hdr">Review Breakdown & Filtering Funnel</div>', unsafe_allow_html=True)
        col_ch1, col_ch2 = st.columns(2)

        df = pd.DataFrame(events)
        df["decision"] = df["review"].apply(lambda r: r["decision"] if r else "unreviewed")
        df["fp_category"] = df["review"].apply(lambda r: (r.get("fp_category") or "unspecified") if (r and r.get("decision") == "rejected") else None)

        with col_ch1:
            st.caption("Human Review Decision Status")
            status_counts = df["decision"].value_counts().reset_index()
            status_counts.columns = ["Decision", "Count"]
            st.bar_chart(status_counts, x="Decision", y="Count", color="#2563eb", height=240)

        with col_ch2:
            st.caption("False Positive Rejection Subtypes")
            fp_df = df[df["fp_category"].notna()]
            if not fp_df.empty:
                fp_counts = fp_df["fp_category"].value_counts().reset_index()
                fp_counts.columns = ["Category", "Count"]
                st.bar_chart(fp_counts, x="Category", y="Count", color="#dc2626", height=240)
            else:
                st.info("No rejected false positives recorded yet to categorize.")

        # ── Registered Videos Table ──
        st.markdown('<div class="section-hdr">Registered Road Videos</div>', unsafe_allow_html=True)
        videos = db.get_videos()
        if videos:
            vdf = pd.DataFrame(videos)
            cols = [c for c in ["video_id", "filename", "processing_status", "duration_sec", "total_frames_sampled", "registered_at"] if c in vdf.columns]
            v_disp = vdf[cols].copy()
            if "duration_sec" in v_disp.columns:
                v_disp["duration_sec"] = v_disp["duration_sec"].apply(lambda v: f"{v:.1f}s" if pd.notna(v) else "—")
            st.dataframe(v_disp, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: PROCESS VIDEO
# ═══════════════════════════════════════════════════════════════════════════
def page_process():
    st.subheader("Process Video")
    st.caption("Analyze road footage, inspect candidate potholes, and compare reporting policies.")
    db = get_db()

    # ── Step 1: Select or Upload Video ──
    st.markdown('<div class="section-hdr">1. Select or Upload Road Video</div>', unsafe_allow_html=True)

    tab_local, tab_upload = st.tabs(["Local Footage", "Upload New Video"])
    video_path_final: Optional[Path] = None
    video_id_final: str = ""

    with tab_local:
        local_videos = []
        if LOCAL_VIDEO_DIR.exists():
            local_videos = sorted(LOCAL_VIDEO_DIR.glob("*.mp4")) + sorted(LOCAL_VIDEO_DIR.glob("*.mov"))

        if local_videos:
            v_choice = st.selectbox(
                "Select existing road video",
                options=[str(p) for p in local_videos],
                format_func=lambda p: Path(p).name,
            )
            v_override = st.text_input("Or enter custom relative/absolute path", value="", key="proc_local_override")
            chosen_path_str = v_override.strip() or v_choice
        else:
            chosen_path_str = st.text_input("Enter video path", placeholder="data/raw/local_video/my_road_clip.mp4")

        if chosen_path_str:
            p = Path(chosen_path_str) if Path(chosen_path_str).is_absolute() else PROJECT_ROOT / chosen_path_str
            if p.exists():
                video_path_final = p
                video_id_final = p.stem.replace(" ", "_")
            else:
                st.error(f"Video file not found: `{p}`")

    with tab_upload:
        uploaded = st.file_uploader("Upload video file (.mp4, .mov)", type=["mp4", "mov", "avi"])
        if uploaded:
            LOCAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            content = uploaded.getvalue()
            upload_name = Path(uploaded.name)
            save_path = LOCAL_VIDEO_DIR / f"{upload_name.stem}_{hashlib.sha256(content).hexdigest()[:12]}{upload_name.suffix.lower()}"
            if not save_path.exists():
                save_path.write_bytes(content)
            video_path_final = save_path
            video_id_final = save_path.stem.replace(" ", "_")
            st.success(f"Video saved to `{save_path.relative_to(PROJECT_ROOT)}`")

    # Display video metadata preview
    if video_path_final and video_path_final.exists():
        meta = get_video_metadata(str(video_path_final))
        if "error" not in meta:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Duration", f"{meta['duration_min']:.1f} min ({meta['duration_sec']:.0f}s)")
            mc2.metric("Resolution", f"{meta['width']} × {meta['height']}")
            mc3.metric("Source FPS", f"{meta['fps']:.1f}")
            mc4.metric("File Size", f"{meta['file_size_mb']:.1f} MB")

            with st.expander("Preview first frame"):
                cap = cv2.VideoCapture(str(video_path_final))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=video_path_final.name, width="stretch")

    # ── Step 2: Select YOLO Checkpoint ──
    st.markdown('<div class="section-hdr">2. Detector & Verifier Checkpoints</div>', unsafe_allow_html=True)

    candidate_models: list[Path] = []
    for folder, pattern in [
        (MODELS_DIR, "**/*.pt"),
        (PROJECT_ROOT / "weights", "**/*.pt"),
        (PROJECT_ROOT / "runs", "**/*.pt"),
        (PROJECT_ROOT, "*.pt"),
    ]:
        if folder.exists():
            candidate_models.extend(folder.glob(pattern))

    def _model_sort_key(p: Path) -> tuple[int, float]:
        is_best = 0 if "best" in p.name.lower() else 1
        mtime = -p.stat().st_mtime if p.exists() else 0.0
        return (is_best, mtime)

    candidate_models = sorted(set(candidate_models), key=_model_sort_key)

    col_ckpt, col_ver = st.columns(2)
    model_valid = False
    model_path_obj: Optional[Path] = None

    with col_ckpt:
        st.markdown("**YOLO Candidate Detector**")
        if candidate_models:
            chosen_ckpt = st.selectbox(
                "Select detector checkpoint",
                options=[str(p) for p in candidate_models],
                format_func=lambda p: Path(p).name,
                help="Place fine-tuned best.pt in models/.",
            )
        else:
            chosen_ckpt = st.text_input("Detector checkpoint path", placeholder="models/best.pt")

        if chosen_ckpt:
            model_path_obj = Path(chosen_ckpt)
            if model_path_obj.exists():
                try:
                    import torch
                    ckpt_data = torch.load(str(model_path_obj), map_location="cpu", weights_only=False)
                    names = None
                    if isinstance(ckpt_data, dict):
                        names = getattr(ckpt_data.get("model"), "names", None) or ckpt_data.get("names")
                    if names and isinstance(names, dict):
                        cls0 = names.get(0, "")
                        if "pothole" in str(cls0).lower():
                            st.success(f"Verified: Class 0 is '{cls0}' ({model_path_obj.stat().st_size / 1e6:.1f} MB)")
                            model_valid = True
                        else:
                            st.error(f"Incompatible Checkpoint: Class 0 is '{cls0}' (not 'pothole'). This base COCO model cannot be run as a pothole detector. Please provide models/best.pt.")
                            model_valid = False
                    else:
                        st.info(f"Loaded checkpoint ({model_path_obj.stat().st_size / 1e6:.1f} MB)")
                        model_valid = True
                except Exception as ex:
                    st.warning(f"Could not inspect checkpoint metadata: {ex}")
                    model_valid = True
            else:
                st.error("Checkpoint file not found.")

    with col_ver:
        st.markdown("**MobileNetV3-Small Verifier (Policy C)**")
        verifier_models = sorted(MODELS_DIR.glob("verifier*.pt")) if MODELS_DIR.exists() else []
        verifier_input = st.text_input(
            "Verifier weights (optional)",
            value=str(verifier_models[0]) if verifier_models else "",
            placeholder="models/verifier_v1.pt",
            help="If not trained yet, leave blank. Policy C will be marked as Unavailable without fabricated scores.",
        )
        verifier_path_final = verifier_input.strip() if (verifier_input.strip() and Path(verifier_input.strip()).exists()) else None
        if verifier_path_final:
            st.success(f"Verifier configured: `{Path(verifier_path_final).name}`")
        else:
            st.info("Verifier not configured. Policy C will be reported as Unavailable.")

    # ── Step 3: Workflows & Visualization ──
    st.markdown('<div class="section-hdr">3. Pipeline & Visualization Workflows</div>', unsafe_allow_html=True)

    tab_annotated, tab_research = st.tabs(["Annotated Video Playback", "Research Event Pipeline"])

    with tab_annotated:
        st.markdown("""
        <div class="info-callout">
            <strong>Annotated Video Visualization:</strong> Evaluates every video frame sequentially with BoT-SORT tracking.
            Renders green bounding boxes, confidence scores, and persistent tracking IDs synchronized to original footage.
        </div>
        """, unsafe_allow_html=True)

        # Check for previously completed runs
        previous_ann_runs = load_annotated_video_records(db, video_id=video_id_final) if video_id_final else []
        active_ann_job_id = st.session_state.get(f"active_annotated_{video_id_final}")

        if previous_ann_runs:
            st.markdown("**Previously Generated Annotated Videos**")
            p_opts = {
                r["job_id"]: f"{r['created_at'][:19].replace('T', ' ')} — {r['total_detections']} detections ({r['model_checkpoint']}, conf={r['detector_conf']:.2f}, {r['output_width']}x{r['output_height']})"
                for r in previous_ann_runs
            }
            default_idx = 0
            if active_ann_job_id and active_ann_job_id in p_opts:
                default_idx = list(p_opts.keys()).index(active_ann_job_id)

            chosen_job_id = st.selectbox(
                "Select completed video playback",
                options=list(p_opts.keys()),
                index=default_idx,
                format_func=lambda j: p_opts[j],
                key=f"sel_prev_ann_{video_id_final}",
            )
            selected_meta = next((r for r in previous_ann_runs if r["job_id"] == chosen_job_id), None)
            if selected_meta:
                _render_annotated_video_view(selected_meta)
            st.divider()

        # Render New Annotated Video Form
        with st.expander("Generate New Annotated Video", expanded=(not previous_ann_runs)):
            st.caption("Perform full-frame inference across the complete recording with bounded memory.")
            av_c1, av_c2 = st.columns(2)
            with av_c1:
                ann_conf = st.slider("Detector Confidence", 0.10, 0.90, 0.35, 0.05, key="ann_conf_slider", help="Minimum confidence required to draw bounding box.")
            with av_c2:
                ann_imgsz = st.selectbox("Inference Resolution (px)", [416, 640, 960, 1280], index=1, key="ann_imgsz_sel")

            av_opt1, av_opt2 = st.columns(2)
            with av_opt1:
                ann_tracking = st.checkbox("Enable BoT-SORT tracking (displays tracking IDs)", value=True, key="ann_track_chk")
            with av_opt2:
                if is_ffmpeg_available():
                    st.caption("FFmpeg is available for H.264 browser streaming.")
                else:
                    st.caption("FFmpeg not detected. Using OpenCV H.264 writer. Install FFmpeg (`brew install ffmpeg`) for optimal web streaming.")

            can_gen_video = model_valid and (video_path_final is not None) and (model_path_obj is not None)
            if not can_gen_video:
                if not model_valid:
                    st.warning("A valid fine-tuned pothole detector checkpoint must be selected.")
                if video_path_final is None:
                    st.warning("Please select or upload a valid road video.")

            if st.button("Generate Annotated Video", icon=":material/movie:", type="primary", disabled=not can_gen_video, width="stretch", key="btn_gen_ann_vid"):
                _execute_annotated_video_pipeline(
                    db=db,
                    video_path=video_path_final,
                    video_id=video_id_final,
                    model_path=model_path_obj,
                    conf_threshold=ann_conf,
                    imgsz=int(ann_imgsz),
                    use_tracking=ann_tracking,
                )

    with tab_research:
        # ── Reprocessing Notice ──
        preserve_reviews = True
        if video_id_final:
            existing_events = db.get_events(video_id=video_id_final, include_reviews=True)
            if existing_events:
                rev_count = sum(1 for e in existing_events if e.get("review"))
                st.markdown(f"""
                <div class="info-callout warning">
                  <strong>Reprocessing Notice:</strong> Video <code>{video_id_final}</code> was previously analyzed
                  ({len(existing_events)} candidate events, {rev_count} human reviews).
                  Re-running will re-segment and update candidate events.
                </div>
                """, unsafe_allow_html=True)
                preserve_reviews = st.checkbox(
                    "Preserve human review decisions on reprocessing (heuristic spatiotemporal re-attachment)",
                    value=True,
                    help="Re-attaches existing human reviews to re-tracked events by spatial bounding-box overlap and timestamp proximity.",
                )

        # ── Step 3: Advanced Settings (Collapsible) ──
        with st.expander("Advanced Inference & Tracking Settings", expanded=False):
            st.caption("Adjust pipeline thresholds. No single threshold is universally optimal; defaults reflect baseline capstone calibration.")
            as1, as2, as3, as4 = st.columns(4)
            with as1:
                sample_fps = st.slider("Sample FPS", 1.0, 15.0, 5.0, 0.5, help="Frame extraction rate for candidate detection.")
            with as2:
                conf_threshold = st.slider("Detector Confidence", 0.10, 0.90, 0.35, 0.05, help="Minimum YOLO box confidence threshold.")
            with as3:
                iou_threshold = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05, help="Non-maximum suppression box overlap.")
            with as4:
                imgsz = st.selectbox("Inference Size (px)", [416, 640, 960, 1280], index=1)

            as5, as6, as7, as8 = st.columns(4)
            with as5:
                policy_b_min_obs = st.number_input("Policy B Min Hits", 1, 10, 2, help="Minimum actual detections required within temporal window.")
            with as6:
                temporal_window = st.number_input("Temporal Window (frames)", 2, 10, 5, help="Sliding window size of sampled frames.")
            with as7:
                verifier_thresh = st.slider("Verifier Threshold (Policy C)", 0.10, 0.90, 0.50, 0.05)
            with as8:
                road_segment_id = st.text_input("Road Segment ID", value="seg_001")

            generate_clips = st.checkbox("Generate short evidence video clips (MP4)", value=False, help="Extracts short 3s MP4 clips around candidate events.")

        # ── Step 4: Execution ──
        st.write("")
        can_execute = model_valid and (video_path_final is not None) and (model_path_obj is not None)

        if not model_valid:
            st.warning("A valid fine-tuned pothole detector checkpoint must be selected before processing.")
        if video_path_final is None:
            st.warning("Please select or upload a valid road video.")

        if st.button("Execute Pipeline", icon=":material/play_arrow:", type="primary", disabled=not can_execute, width="stretch"):
            _execute_video_pipeline(
                db=db,
                video_path=video_path_final,
                video_id=video_id_final,
                model_path=model_path_obj,
                verifier_path=verifier_path_final,
                sample_fps=sample_fps,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                imgsz=int(imgsz),
                road_segment_id=road_segment_id,
                policy_b_min_obs=int(policy_b_min_obs),
                policy_b_window=int(temporal_window),
                verifier_threshold=verifier_thresh,
                preserve_reviews=preserve_reviews,
                generate_clips=generate_clips,
            )

        # ── Database Maintenance / Cleanup ──
        with st.expander("Database Maintenance & Danger Zone", expanded=False):
            st.caption("Destructive actions are kept separate from standard workflow and require explicit confirmation.")
            col_cl1, col_cl2 = st.columns([3, 1])
            with col_cl1:
                confirm_clear = st.checkbox("I understand this will permanently delete all candidate events, reviews, and disk evidence for this video.", key="chk_clear_vid")
            with col_cl2:
                if st.button("Delete Video Events", icon=":material/delete:", type="secondary", disabled=not confirm_clear, width="stretch"):
                    if video_id_final:
                        if hasattr(db, "delete_video_data"):
                            db.delete_video_data(video_id_final, clear_reviews=True, reset_status=True)
                        else:
                            db.delete_events_for_video(video_id_final, clear_reviews=True)
                        n_f, sz_mb = delete_evidence_disk(video_id_final)
                        st.success(f"Cleared events and {n_f} evidence files ({sz_mb:.1f} MB) for `{video_id_final}`.")
                        st.rerun()


def _render_annotated_video_view(meta_dict: dict[str, Any]):
    output_path = Path(meta_dict["output_video_path"])
    if not output_path.exists():
        st.error(f"Annotated video file not found on disk: `{output_path}`")
        return

    st.markdown("#### Annotated Video Playback")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">FRAMES EVALUATED</div>
            <div class="kpi-value blue">{meta_dict['processed_frame_count']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">EFFECTIVE SPEED</div>
            <div class="kpi-value green">{meta_dict['effective_fps']:.1f} <span style="font-size:1rem;">FPS</span></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">DETECTIONS DRAWN</div>
            <div class="kpi-value amber">{meta_dict['total_detections']}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">UNIQUE TRACKS</div>
            <div class="kpi-value slate">{meta_dict['unique_tracks']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.video(str(output_path))
    st.caption(
        f"Playback: {meta_dict.get('output_width', 0)}x{meta_dict.get('output_height', 0)} @ {meta_dict.get('output_fps', 0):.1f} FPS "
        f"— Source: {Path(meta_dict.get('input_video_path', '')).name} — Checkpoint: {meta_dict.get('model_checkpoint', '')} "
        f"— Inference: {meta_dict.get('processing_duration', 0):.1f}s ({meta_dict.get('effective_fps', 0):.1f} FPS)"
    )

    col_dl, col_blank = st.columns([1, 2])
    with col_dl:
        try:
            with open(output_path, "rb") as vf:
                video_bytes = vf.read()
            st.download_button(
                label="Download Annotated Video (MP4)",
                data=video_bytes,
                file_name=output_path.name,
                mime="video/mp4",
                icon=":material/download:",
                type="primary",
                width="stretch",
            )
        except Exception as exc:
            st.warning(f"Could not prepare video download: {exc}")

    with st.expander("Annotated Video Job Metadata & Technical Details", expanded=False):
        meta_json = meta_dict.get("metadata_json")
        if meta_json:
            try:
                st.json(json.loads(meta_json))
            except Exception:
                st.json(meta_dict)
        else:
            st.json(meta_dict)


def _execute_annotated_video_pipeline(
    db: Database,
    video_path: Path,
    video_id: str,
    model_path: Path,
    conf_threshold: float,
    imgsz: int,
    use_tracking: bool,
):
    from src.video.annotated_renderer import AnnotatedVideoRenderer, RenderProgress

    renderer = AnnotatedVideoRenderer(output_dir=ANNOTATED_VIDEOS_DIR)
    device = auto_device()

    with st.status("Rendering Annotated Video...", expanded=True) as status:
        st.write(f"1. Loading detector `{model_path.name}` on **{device.upper()}**...")
        prog_bar = st.progress(0.0)
        prog_text = st.empty()

        def on_prog(p: RenderProgress):
            prog_bar.progress(p.percent / 100.0)
            eta_txt = f", ETA: {p.eta_sec:.1f}s" if p.eta_sec is not None else ""
            prog_text.text(
                f"Processing frame {p.current_frame}/{p.total_frames} ({p.percent:.0f}%) — "
                f"Speed: {p.current_fps:.1f} FPS, Elapsed: {p.elapsed_sec:.1f}s{eta_txt}"
            )

        st.write("2. Sequential frame-by-frame inference & annotation...")
        try:
            meta = renderer.render_annotated_video(
                video_path=video_path,
                model=model_path,
                conf_threshold=conf_threshold,
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

        st.write("3. Persisting rendering metadata to database...")
        if hasattr(db, "save_annotated_video"):
            db.save_annotated_video(meta)

        # Register video in db if not already registered
        if not db.get_video(video_id):
            meta_dict = get_video_metadata(str(video_path))
            meta_dict["filepath"] = str(video_path)
            meta_dict["road_segment_id"] = "seg_001"
            db.register_video(video_id, meta_dict)

        status.update(
            label=f"Completed in {meta.processing_duration_sec:.1f}s! ({meta.processed_frame_count} frames, {meta.total_detections_drawn} detections)",
            state="complete",
            expanded=False,
        )

    st.session_state[f"active_annotated_{video_id}"] = meta.job_id
    st.success(
        f"Annotated video generated successfully: {meta.processed_frame_count} frames, "
        f"{meta.total_detections_drawn} detections in {meta.processing_duration_sec:.1f}s ({meta.effective_inference_fps:.1f} FPS)."
    )
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
            message = f"Results were saved, but the completion display failed: {exc}. Reopen Review Events."
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

    with st.status("Processing Road Video...", expanded=True) as status:
        st.write(f"1. Loading detector checkpoint `{model_path.name}` on **{device.upper()}**...")
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
        st.write(f"2. Incremental detection & tracking ({total_sec:.1f}s video at {sample_fps} FPS, ~{expected} frames)...")

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
        st.write(f"BoT-SORT grouped detections into **{len(events)} candidate events**.")

        # Verifier scoring
        verifier = load_verifier(verifier_path, threshold=verifier_threshold)
        if verifier.is_configured():
            st.write("3. Running MobileNetV3-Small visual verification (Policy C)...")
            for ev in events:
                if ev.crop_path and Path(ev.crop_path).exists():
                    cbgr = cv2.imread(ev.crop_path)
                    if cbgr is not None:
                        res = verifier.score_crop(cbgr)
                        ev.policy_c_score = res.pothole_score
                        ev.policy_c_retained = res.is_pothole
        else:
            st.write("3. MobileNetV3 verifier not configured — Policy C marked as Unavailable.")
            for ev in events:
                ev.policy_c_score = None
                ev.policy_c_retained = None

        # Reporting policies
        st.write("4. Applying reporting suppression policies (A, B, C)...")
        pcfg = PolicyConfig(
            conf_threshold=conf_threshold,
            min_obs=policy_b_min_obs,
            temporal_window_frames=policy_b_window,
            verifier_threshold=verifier_threshold,
            policy_c_requires_policy_b=False,
        )
        apply_policies(events, pcfg)

        # Persist
        st.write("5. Persisting events to SQLite database...")
        run_config["tracker"] = events[0].tracker_config if events else grouper._tracker_config_snapshot()
        run_config["status"] = "complete"
        transfer = db.replace_completed_run(video_id, meta, events, run_config,
                                            actual_frames[0], preserve_reviews)
        if preserve_reviews:
            st.write(f"Reviews transferred heuristically: {transfer['transferred']}; unmatched: {transfer['unmatched']}. Previous reviews remain in the run archive.")
        elapsed_total = time.time() - t0
        status.update(label=f"Completed in {elapsed_total:.1f}s! ({len(events)} events generated)", state="complete", expanded=False)

    st.success(f"Processing finished in {elapsed_total:.1f}s. Generated {len(events)} candidate events.")
    sum_res = summarize_policies(events, pcfg)
    m1, m2, m3 = st.columns(3)
    m1.metric("Policy A (Detector Only)", f"{sum_res['policy_a_retained']} / {sum_res['total_candidates']}")
    m2.metric("Policy B (Temporal Window)", f"{sum_res['policy_b_retained']} / {sum_res['total_candidates']}")
    m3.metric(
        "Policy C (Visual Verifier)",
        f"{sum_res['policy_c_retained']} / {sum_res['policy_c_scored']}" if verifier.is_configured() else "Unavailable",
    )

    if st.button("Proceed to Review Events", icon=":material/fact_check:", type="primary"):
        st.session_state["pending_nav_page"] = "Review Events"
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: REVIEW EVENTS (MASTER/DETAIL WORKFLOW)
# ═══════════════════════════════════════════════════════════════════════════
def page_review_events():
    st.subheader("Review Candidate Events")
    db = get_db()

    # ── Top Filter Bar ──
    f1, f2, f3, f4, f5 = st.columns(5)
    videos = db.get_videos()
    vid_options = ["All videos"] + [v["video_id"] for v in videos]
    with f1:
        sel_video = st.selectbox("Video", vid_options, key="rev_filter_vid")
    with f2:
        sel_rev_status = st.selectbox("Review Status", ["All", "Unreviewed", "Confirmed", "Rejected", "Uncertain", "Flag Grouping"], key="rev_filter_status")
    with f3:
        sel_policy = st.selectbox(
            "Policy Filter",
            ["All candidates", "Policy A retained", "Policy B retained", "Policy C retained", "Filtered by Policy A", "Filtered by Policy B", "Filtered by Policy C", "Policy C unavailable"],
            key="rev_filter_policy",
        )
    with f4:
        sel_fp_cat = st.selectbox("Rejection Type", ["All", "shadow", "repaired_patch", "manhole", "ordinary_road", "other"], key="rev_filter_fp")
    with f5:
        sel_sort = st.selectbox("Sort Order", ["Time (earliest)", "Time (latest)", "Confidence (highest)", "Observations (most)"], key="rev_filter_sort")

    search_id = st.text_input("Search Event ID", placeholder="Filter by event ID...", key="rev_search_id").strip()

    st.markdown("""
    <div style="font-size:0.80rem;color:#64748b;margin-bottom:12px;">
      <strong>Audit Note:</strong> Filtered-out candidates remain accessible here for false-rejection audits.
      Rejecting a candidate is a human review decision, distinct from automated model suppression.
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch and Filter Events ──
    vid_filter_arg = None if sel_video == "All videos" else sel_video
    all_events = db.get_events(video_id=vid_filter_arg, include_reviews=True)

    segments = sorted({e["road_segment_id"] for e in all_events})
    segment = st.selectbox("Road segment", ["All segments"] + segments, key="rev_segment")
    filtered_events = [e for e in all_events if segment == "All segments" or e["road_segment_id"] == segment]

    if search_id:
        filtered_events = [e for e in filtered_events if search_id.lower() in e["event_id"].lower()]

    if sel_policy == "Policy A retained":
        filtered_events = [e for e in filtered_events if e.get("policy_a_retained")]
    elif sel_policy == "Policy B retained":
        filtered_events = [e for e in filtered_events if e.get("policy_b_retained")]
    elif sel_policy == "Policy C retained":
        filtered_events = [e for e in filtered_events if e.get("policy_c_retained")]
    elif sel_policy == "Filtered by Policy A":
        filtered_events = [e for e in filtered_events if not e.get("policy_a_retained")]
    elif sel_policy == "Filtered by Policy B":
        filtered_events = [e for e in filtered_events if not e.get("policy_b_retained")]

    elif sel_policy == "Filtered by Policy C":
        filtered_events = [e for e in filtered_events if e.get("policy_c_retained") == 0]
    elif sel_policy == "Policy C unavailable":
        filtered_events = [e for e in filtered_events if e.get("policy_c_retained") is None]

    if sel_rev_status != "All":
        target = sel_rev_status.lower().replace(" ", "_")
        if target == "unreviewed":
            filtered_events = [e for e in filtered_events if e.get("review") is None]
        else:
            filtered_events = [e for e in filtered_events if e.get("review") and e["review"]["decision"] == target]

    if sel_fp_cat != "All":
        filtered_events = [
            e for e in filtered_events
            if e.get("review") and (e["review"].get("fp_category") or "").lower() == sel_fp_cat.lower()
        ]

    # Sorting
    if sel_sort.startswith("Time (earliest)"):
        filtered_events.sort(key=lambda e: e.get("first_timestamp_ms") or 0)
    elif sel_sort.startswith("Time (latest)"):
        filtered_events.sort(key=lambda e: -(e.get("first_timestamp_ms") or 0))
    elif sel_sort.startswith("Confidence"):
        filtered_events.sort(key=lambda e: -(e.get("detector_conf_max") or 0.0))
    elif sel_sort.startswith("Observations"):
        filtered_events.sort(key=lambda e: -(e.get("observation_count") or 0))

    if not filtered_events:
        st.info("No candidate events match the current filter selection.")
        return

    # Keep selected event ID valid
    valid_ids = [e["event_id"] for e in filtered_events]
    if st.session_state["selected_event_id"] not in valid_ids:
        st.session_state["selected_event_id"] = valid_ids[0]

    # ── Master / Detail Layout ──
    col_master, col_detail = st.columns([4, 8])

    with col_master:
        st.markdown(f"**Candidate Events ({len(filtered_events)})**")

        # Compact selection list
        selected_index = valid_ids.index(st.session_state["selected_event_id"])

        def format_event_label(e: dict) -> str:
            eid = e["event_id"]
            ts = fmt_ts(e.get("first_timestamp_ms", 0))
            c = e.get("detector_conf_max", 0.0) or 0.0
            n = e.get("observation_count", 0)
            rev = (e.get("review") or {}).get("decision", "unreviewed").upper()
            pb = "B:PASS" if e.get("policy_b_retained") else "B:FILT"
            return f"[{rev[:4]}] {ts} · conf {c:.2f} · {n} obs · {pb} · {eid}"

        event_map = {e["event_id"]: e for e in filtered_events}

        selected_id = st.radio(
            "Select event to review",
            options=valid_ids,
            format_func=lambda eid: format_event_label(event_map[eid]),
            label_visibility="collapsed",
            key="selected_event_id",
        )
        # Radio is bound directly to selected_event_id.

    # ── Detail Panel ──
    with col_detail:
        cur_event = event_map[st.session_state["selected_event_id"]]
        _render_event_detail_panel(cur_event, db, filtered_events)


def _render_event_detail_panel(ev: dict, db: Database, candidate_list: list[dict]):
    eid = ev["event_id"]
    if st.session_state.get("pending_clear_event") == eid:
        for prefix in ("widget_dec_", "widget_fp_", "widget_notes_"):
            st.session_state.pop(prefix + eid, None)
        st.session_state.pop("pending_clear_event", None)
    vid = ev["video_id"]
    review = ev.get("review") or {}
    cur_dec = review.get("decision", "unreviewed")
    cur_notes = review.get("reviewer_notes", "") or ""
    cur_fp = review.get("fp_category", "") or ""

    # Navigation buttons
    cur_idx = [e["event_id"] for e in candidate_list].index(eid)
    col_hdr, col_nav = st.columns([5, 3])
    with col_hdr:
        st.markdown(
            f"<h4 style='margin:0;display:inline;'>Event <code>{eid}</code></h4> &nbsp; {render_decision_badge(cur_dec)}",
            unsafe_allow_html=True,
        )
    with col_nav:
        nc1, nc2 = st.columns(2)
        with nc1:
            if st.button("Previous", icon=":material/chevron_left:", disabled=(cur_idx == 0), width="stretch", key=f"btn_prev_{eid}"):
                st.session_state["pending_selected_event_id"] = candidate_list[cur_idx - 1]["event_id"]
                st.rerun()
        with nc2:
            if st.button("Next", icon=":material/chevron_right:", disabled=(cur_idx >= len(candidate_list) - 1), width="stretch", key=f"btn_next_{eid}"):
                st.session_state["pending_selected_event_id"] = candidate_list[cur_idx + 1]["event_id"]
                st.rerun()

    st.write("")

    # ── Side-by-Side Evidence ──
    col_crop, col_frame = st.columns([1, 1])

    crop_path = ev.get("crop_path")
    full_path = ev.get("full_frame_path")

    with col_crop:
        st.caption("Selected evidence crop")
        if crop_path and Path(crop_path).exists():
            st.image(crop_path, width="stretch")
        else:
            st.info("No crop image file found.")

    with col_frame:
        st.caption("Full frame context")
        if full_path and Path(full_path).exists():
            st.image(full_path, width="stretch")
        else:
            st.info("No full-frame image file found.")

    # ── Synced Video Playback ──
    vrow = db.get_video(vid)
    if vrow and vrow.get("filepath") and Path(vrow["filepath"]).exists():
        with st.expander("Source Video Playback (synced to evidence timestamp)", expanded=False):
            rep_ts = ev.get("rep_timestamp_ms", ev.get("first_timestamp_ms", 0))
            start_sec = max(0, (rep_ts // 1000) - 2)
            st.video(str(vrow["filepath"]), start_time=start_sec)
            st.caption(f"Playing from ~{start_sec}s (2s before observation at {fmt_ts_full(rep_ts)}).")

    # ── Spatiotemporal & Model Metadata ──
    st.markdown('<div class="section-hdr">Event details</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.caption("Observation Interval")
        st.write(f"{fmt_ts_full(ev['first_timestamp_ms'])} → {fmt_ts_full(ev['last_timestamp_ms'])}")
        dur_s = (ev.get("duration_ms", 0) or 0) / 1000.0
        st.caption(f"Duration: {dur_s:.2f}s ({ev['observation_count']} obs)")
    with m2:
        st.caption("Representative Observation")
        st.write(f"Timestamp: **{fmt_ts_full(ev.get('rep_timestamp_ms', ev['first_timestamp_ms']))}**")
        st.caption(f"Frame index: {ev.get('rep_frame_index', '—')}")
    with m3:
        st.caption("Detector Confidence")
        cmax = ev.get("detector_conf_max") or 0.0
        cmean = ev.get("detector_conf_mean") or 0.0
        st.write(f"Max: **{cmax:.3f}** | Mean: {cmean:.3f}")
        st.caption("Raw detector score (uncalibrated)")
    with m4:
        st.caption("Verifier Score (Policy C)")
        pscore = ev.get("policy_c_score")
        if pscore is not None:
            st.write(f"Score: **{pscore:.3f}**")
        else:
            st.write("**Unavailable**")
            st.caption("Verifier not evaluated")

    config = json.loads((vrow or {}).get("run_config") or "{}")
    unknown = "Unknown—legacy run"
    # ── Policy Outcome Explanations ──
    st.markdown('<div class="section-hdr">Reporting Policies Breakdown</div>', unsafe_allow_html=True)
    p_a_html = render_policy_badge(bool(ev.get("policy_a_retained")), "A — Detector Only")
    p_b_html = render_policy_badge(bool(ev.get("policy_b_retained")), "B — Temporal Check")
    pc_ret = ev.get("policy_c_retained")
    p_c_html = render_policy_badge(bool(pc_ret) if pc_ret is not None else None, "C — Visual Verifier", ev.get("policy_c_score"))

    st.markdown(f"{p_a_html} &nbsp; {p_b_html} &nbsp; {p_c_html}", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.83rem;color:inherit;margin-top:6px;">
      • <strong>Policy A:</strong> Detector confidence {cmax:.3f}; threshold {config.get("detector_conf", unknown)} &rarr; {'PASS' if ev.get('policy_a_retained') else 'FILTERED'}<br/>
      • <strong>Policy B:</strong> {config.get("min_obs", unknown)} required detections within {config.get("temporal_window_frames", unknown)} sampled frames &rarr; {'PASS' if ev.get('policy_b_retained') else 'FILTERED'} ({ev['observation_count']} observations)<br/>
      • <strong>Policy C:</strong> {'Score ' + f'{pscore:.3f}' if pscore is not None else 'Unavailable (no valid score)'} &rarr; {'PASS' if ev.get('policy_c_retained') else 'FILTERED/UNAVAILABLE'}
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Verifier threshold: {config.get('verifier_threshold', unknown)} · Run: {config.get('run_id', unknown)}")

    # ── Human Review Controls ──
    st.markdown('<div class="section-hdr">Human Review Decision</div>', unsafe_allow_html=True)

    with st.container():
        decision_options = ["unreviewed", "confirmed", "rejected", "uncertain", "flag_grouping"]
        dec_idx = decision_options.index(cur_dec) if cur_dec in decision_options else 0

        col_dec, col_cat = st.columns([1, 1])
        with col_dec:
            new_decision = st.selectbox(
                "Review Decision",
                options=decision_options,
                index=dec_idx,
                format_func=lambda s: {
                    "unreviewed": "Unreviewed (pending)",
                    "confirmed": "Confirm Pothole (True Positive)",
                    "rejected": "Reject (False Alarm)",
                    "uncertain": "Uncertain (exclude from scored eval)",
                    "flag_grouping": "Flag Grouping Error (tracker issue)",
                }.get(s, s),
                key=f"widget_dec_{eid}",
            )

        with col_cat:
            fp_options = ["", "shadow", "repaired_patch", "manhole", "ordinary_road", "other"]
            fp_idx = fp_options.index(cur_fp) if cur_fp in fp_options else 0
            new_fp = st.selectbox(
                "Rejection category",
                options=fp_options,
                index=fp_idx,
                disabled=(new_decision != "rejected"),
                format_func=lambda s: s if s else "— Select reason —",
                key=f"widget_fp_{eid}",
            )

        # Notes
        notes_key = f"notes_{eid}"
        new_notes = st.text_area(
            "Reviewer Notes",
            value=st.session_state["review_notes_draft"].get(eid, cur_notes),
            height=70,
            key=f"widget_notes_{eid}",
        )
        st.session_state["review_notes_draft"][eid] = new_notes

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            if st.button("Save Decision", icon=":material/check:", type="primary", width="stretch", key=f"btn_save_{eid}"):
                if new_decision == "unreviewed":
                    db.delete_review(eid)
                else:
                    db.save_review(eid, new_decision, (new_fp or None) if new_decision == "rejected" else None, new_notes)
                st.session_state["review_saved_msg"] = f"Saved decision for {eid}."
                st.rerun()

        with rc2:
            if st.button("Save & Next Unreviewed", icon=":material/playlist_add_check:", width="stretch", key=f"btn_save_next_{eid}"):
                if new_decision == "unreviewed":
                    db.delete_review(eid)
                else:
                    db.save_review(eid, new_decision, (new_fp or None) if new_decision == "rejected" else None, new_notes)

                # Find next unreviewed candidate
                all_evs = db.get_events(video_id=vid, include_reviews=True)
                next_unrev = [e["event_id"] for e in all_evs if e.get("review") is None and e["event_id"] != eid]
                if next_unrev:
                    st.session_state["pending_selected_event_id"] = next_unrev[0]
                st.session_state["review_saved_msg"] = f"Saved {eid} and advanced."
                st.rerun()

        with rc3:
            if st.button("Clear Review", icon=":material/undo:", width="stretch", key=f"btn_clear_{eid}"):
                db.delete_review(eid)
                st.session_state["pending_clear_event"] = eid
                st.session_state["review_notes_draft"].pop(eid, None)
                st.session_state["review_saved_msg"] = f"Cleared review for {eid}."
                st.rerun()

    if st.session_state.get("review_saved_msg"):
        st.toast(st.session_state["review_saved_msg"])
        st.session_state["review_saved_msg"] = None


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: EVALUATION (HONEST RESEARCH-GRADE ANALYSIS)
# ═══════════════════════════════════════════════════════════════════════════
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
        st.info("No videos registered. Go to **Process Video** to analyze a video first.")
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
    st.subheader("Export Results")
    st.caption("Download structured research CSV reports containing all model predictions, timestamps, evidence paths, and human review decisions.")
    db = get_db()

    st.download_button("Download previous run archive", get_db().export_run_archive(),
                       "run_archive.json", "application/json", icon=":material/history:")
    tab_report, tab_crops = st.tabs(["Event Report Export", "Verifier Crop Inventory Export"])

    with tab_report:
        videos = db.get_videos()
        vid_opts = ["All videos"] + [v["video_id"] for v in videos]

        c_v, c_rev, c_pol = st.columns(3)
        with c_v:
            sel_exp_vid = st.selectbox("Export scope", vid_opts, key="exp_sel_vid")
        with c_rev:
            sel_exp_rev = st.selectbox("Review filter", ["All events", "Reviewed events only", "Unreviewed only"], key="exp_sel_rev")
        with c_pol:
            sel_exp_pol = st.selectbox("Policy filter", ["All proposed candidates", "Policy A retained", "Policy B retained", "Policy C retained"], key="exp_sel_pol")

        vid_arg = None if sel_exp_vid == "All videos" else sel_exp_vid
        events = db.get_events(video_id=vid_arg, include_reviews=True)

        # Filter
        if sel_exp_rev == "Reviewed events only":
            events = [e for e in events if e.get("review")]
        elif sel_exp_rev == "Unreviewed only":
            events = [e for e in events if not e.get("review")]

        if sel_exp_pol == "Policy A retained":
            events = [e for e in events if e.get("policy_a_retained")]
        elif sel_exp_pol == "Policy B retained":
            events = [e for e in events if e.get("policy_b_retained")]
        elif sel_exp_pol == "Policy C retained":
            events = [e for e in events if e.get("policy_c_retained")]

        st.caption(f"**{len(events)} events** matching export criteria.")

        if events:
            # Build CSV string
            videos_meta = {v["video_id"]: v for v in db.get_videos()}
            fields = [
                "event_id", "video_id", "road_segment_id", "track_id",
                "first_timestamp_ms", "last_timestamp_ms", "duration_ms",
                "first_frame_index", "last_frame_index",
                "observation_count", "detector_conf_max", "detector_conf_mean",
                "rep_x1", "rep_y1", "rep_x2", "rep_y2",
                "rep_timestamp_ms", "rep_frame_index",
                "policy_a_retained", "policy_b_retained",
                "policy_c_score", "policy_c_retained",
                "full_frame_path", "crop_path", "clip_path",
                "review_decision", "fp_category", "reviewer_notes", "reviewed_at",
                "model_path", "model_version", "sample_fps", "detector_conf", "run_config", "created_at",
            ]

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()

            for ev in events:
                rev = ev.get("review") or {}
                vmeta = videos_meta.get(ev.get("video_id", ""), {})
                row = {
                    **ev,
                    "review_decision": rev.get("decision", "unreviewed"),
                    "fp_category": rev.get("fp_category", ""),
                    "reviewer_notes": rev.get("reviewer_notes", ""),
                    "reviewed_at": rev.get("reviewed_at", ""),
                    "run_config": vmeta.get("run_config", ""),
                    "model_path": vmeta.get("model_path", ""),
                    "model_version": vmeta.get("model_version", ""),
                    "sample_fps": vmeta.get("sample_fps", ""),
                    "detector_conf": vmeta.get("detector_conf", ""),
                }
                writer.writerow(row)

            csv_text = buf.getvalue()
            filename = f"pothole_events_{sel_exp_vid.replace(' ', '_')}.csv"

            st.download_button(
                label="Download Event Report CSV",
                data=csv_text,
                file_name=filename,
                mime="text/csv",
                type="primary",
                width="stretch",
            )

            st.write("")
            preview_df = pd.read_csv(io.StringIO(csv_text))
            st.dataframe(preview_df.head(25), width="stretch", hide_index=True)
        else:
            st.info("No events match the selected criteria.")

    with tab_crops:
        st.markdown('<div class="section-hdr">Export Verifier Crop Training Inventory</div>', unsafe_allow_html=True)
        st.caption("Exports candidate crops classified by human review (confirmed = pothole, rejected = non_pothole with look-alike subtype) to train MobileNetV3-Small.")

        all_events = db.get_events(include_reviews=True)
        confirmed_crops = [e for e in all_events if e.get("review") and e["review"]["decision"] == "confirmed" and e.get("crop_path")]
        rejected_crops = [e for e in all_events if e.get("review") and e["review"]["decision"] == "rejected" and e.get("crop_path")]

        cr1, cr2 = st.columns(2)
        cr1.metric("Confirmed Pothole Crops (Positive)", len(confirmed_crops))
        cr2.metric("Rejected Look-Alike Crops (Negative)", len(rejected_crops))

        if confirmed_crops or rejected_crops:
            cbuf = io.StringIO()
            cwriter = csv.DictWriter(cbuf, fieldnames=["event_id", "video_id", "label", "subtype", "crop_path", "detector_conf_max", "rep_timestamp_ms"])
            cwriter.writeheader()

            for ev in confirmed_crops:
                cwriter.writerow({
                    "event_id": ev["event_id"], "video_id": ev["video_id"],
                    "label": "pothole", "subtype": "pothole",
                    "crop_path": ev.get("crop_path", ""),
                    "detector_conf_max": ev.get("detector_conf_max", 0.0),
                    "rep_timestamp_ms": ev.get("rep_timestamp_ms", ev.get("first_timestamp_ms", 0)),
                })
            for ev in rejected_crops:
                rv = ev.get("review") or {}
                cwriter.writerow({
                    "event_id": ev["event_id"], "video_id": ev["video_id"],
                    "label": "non_pothole", "subtype": rv.get("fp_category", "ordinary_road") or "ordinary_road",
                    "crop_path": ev.get("crop_path", ""),
                    "detector_conf_max": ev.get("detector_conf_max", 0.0),
                    "rep_timestamp_ms": ev.get("rep_timestamp_ms", ev.get("first_timestamp_ms", 0)),
                })

            st.download_button(
                "Download Crop Manifest CSV",
                data=cbuf.getvalue(),
                file_name="reviewed_crop_inventory.csv",
                mime="text/csv",
                type="primary",
            )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN & SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════
PAGES = {
    "Overview": page_overview,
    "Process Video": page_process,
    "Review Events": page_review_events,
    "Evaluation": page_evaluation,
    "Export": page_export,
}


def main():
    for key in ("nav_page", "selected_event_id"):
        pending = st.session_state.pop(f"pending_{key}", None)
        if pending is not None:
            st.session_state[key] = pending
    with st.sidebar:
        st.markdown("### RoadLens")
        st.caption("POTHOLE VERIFICATION")
        st.caption("Detection · Verification · Review")
        st.divider()

        # Navigation
        page_names = list(PAGES.keys())
        if st.session_state.get("nav_page") not in page_names:
            st.session_state["nav_page"] = page_names[0]

        selected_page = st.radio(
            "Primary Navigation",
            options=page_names,
            key="nav_page",
            label_visibility="collapsed",
            format_func=lambda page: {"Overview": ":material/dashboard: Overview", "Process Video": ":material/video_file: Process Video", "Review Events": ":material/fact_check: Review Events", "Evaluation": ":material/analytics: Evaluation", "Export": ":material/download: Export"}[page],
        )
        # Radio is bound directly to nav_page.

        st.divider()

        # Database Status
        db = get_db()
        stats = db.get_summary_stats()
        st.caption(
            f"**Database:** `{DB_PATH.name}`\n\n"
            f"**Videos:** {stats['total_videos']} &nbsp;|&nbsp; **Events:** {stats['total_events']}\n\n"
            f"**Reviewed:** {stats['reviewed']} / {stats['total_events']}"
        )

        st.divider()

        # Annotation Tool Access Callout
        st.markdown("""
        <div style="font-size:0.78rem;color:#64748b;line-height:1.4;">
          <strong>Ground Truth Annotation Tool:</strong><br/>
          To annotate raw training frames and create evaluation ground truth, run in terminal:<br/>
          <code>streamlit run dashboard/annotate.py</code>
        </div>
        """, unsafe_allow_html=True)

    # Render selected page
    PAGES[st.session_state["nav_page"]]()


if __name__ == "__main__":
    main()
