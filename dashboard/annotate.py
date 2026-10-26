"""Pothole annotation interface — Streamlit application.

Features:
- Stable drag-and-draw canvas: boxes stay locked and do not disappear.
- Interactive Zoom controls: 1.0x to 2.5x magnification with scrollable inspection viewport.
- Editable boxes after creation: adjust category, coordinates, and size directly in the inspector table.
- Transform mode: click any box on canvas to move, stretch, or resize.
- Native theme compatibility: zero white-on-white text in both Dark and Light modes.
- Fullscreen high-resolution inspection modal.
- Single-click review actions and auto-advance for rapid labeling.
- 100% professional styling with zero emojis.

Launch from project root:
    streamlit run dashboard/annotate.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:
    sys.exit(
        "ERROR: Streamlit is not installed.\n"
        "  Install with: .venv/bin/pip install streamlit\n"
        "  Then run: streamlit run dashboard/annotate.py"
    )

try:
    from PIL import Image, ImageDraw
except ImportError:
    st.error("Pillow is required: .venv/bin/pip install Pillow")
    st.stop()

# ---------------------------------------------------------------------------
# Interactive Drawing Canvas Initialization
# ---------------------------------------------------------------------------
CANVAS_AVAILABLE = False
try:
    from streamlit.components.v2.component_manager import BidiComponentManager
    
    # Ensure any BidiComponentManager instance auto-discovers package component manifests
    _orig_mgr_init = BidiComponentManager.__init__
    def _auto_discover_init(self, *args, **kwargs):
        _orig_mgr_init(self, *args, **kwargs)
        try:
            self.discover_and_register_components()
        except Exception:
            pass
    BidiComponentManager.__init__ = _auto_discover_init

    from streamlit.runtime import Runtime
    if Runtime.exists():
        try:
            Runtime.instance().bidi_component_registry.discover_and_register_components()
        except Exception:
            pass

    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project paths & imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src import annotations as ann_io
except ImportError as err:
    st.error(f"Cannot import src/annotations.py: {err}. Please run from the project root.")
    st.stop()

INVENTORY_PATH = PROJECT_ROOT / "data" / "metadata" / "image_inventory.csv"
VIDEO_FRAMES_ROOT = PROJECT_ROOT / "data" / "processed" / "video_frames"

# ---------------------------------------------------------------------------
# Page configuration & Clean Theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pothole Verification & Annotation Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Streamlit padding overrides for zero-scroll viewport */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        height: 2.2rem !important;
        background: transparent !important;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Top Navigation Banner using Streamlit Native Theme Variables */
    .top-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.85rem;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.08));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        margin-bottom: 0.75rem;
        color: var(--text-color, inherit);
    }
    .top-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--text-color, inherit);
        margin: 0;
    }
    .top-meta {
        font-size: 0.8rem;
        color: var(--text-color, inherit);
        opacity: 0.75;
    }

    /* Status badge pill */
    .status-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-radius: 9999px;
    }
    .status-reviewed {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .status-unreviewed {
        background-color: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    .status-uncertain {
        background-color: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }

    /* Controls and radio group spacing */
    div[data-testid="stRadio"] {
        margin-bottom: 0.6rem !important;
    }
    div[data-testid="stRadio"] > div {
        gap: 0.75rem !important;
    }
    div[data-testid="stHorizontalBlock"] {
        margin-bottom: 0.6rem !important;
        align-items: center !important;
    }

    /* Button styling and spacing to guarantee clearance from borders */
    div[data-testid="stButton"] > button {
        padding: 0.35rem 0.5rem !important;
        font-size: 0.85rem !important;
        min-height: 2.2rem !important;
        line-height: 1.2 !important;
        margin-bottom: 0.25rem !important;
        border-radius: 6px !important;
        white-space: nowrap !important;
    }

    /* Canvas container border wrapper: ample margin-top and inner padding */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: 1.25rem !important;
        margin-bottom: 1.25rem !important;
        padding: 12px !important;
        border: 1.5px solid rgba(128, 128, 128, 0.28) !important;
        border-radius: 8px !important;
        background: #090d16 !important;
    }

    /* Box inspector card */
    .box-card {
        padding: 0.5rem 0.75rem;
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.06));
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers & Data IO
# ---------------------------------------------------------------------------
def rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


@st.cache_data(ttl=30)
def load_inventory() -> list[dict]:
    if not INVENTORY_PATH.exists():
        return []
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("image_id")]


def update_inventory_record(image_id: str, updates: dict) -> None:
    """Synchronize annotation state back to image_inventory.csv."""
    if not INVENTORY_PATH.exists():
        return
    rows = []
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for r in reader:
            if r.get("image_id") == image_id:
                r.update(updates)
            rows.append(r)
    with INVENTORY_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_video_frame_index() -> dict[str, dict]:
    index: dict[str, dict] = {}
    for frames_csv in VIDEO_FRAMES_ROOT.rglob("frames.csv"):
        with frames_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                fp = row.get("filepath", "")
                if fp:
                    index[fp] = {
                        "timestamp_sec": row.get("timestamp_sec", ""),
                        "video_id": row.get("video_id", ""),
                    }
    return index


def save_annotation_from_state(image_id: str, inv_row: dict, state: dict) -> Path:
    img_path = PROJECT_ROOT / inv_row["filepath"]
    try:
        with Image.open(img_path) as img:
            w, h = img.size
    except Exception:
        w = int(inv_row.get("width", 0))
        h = int(inv_row.get("height", 0))

    ann: dict = ann_io.load(image_id)
    ann["source_filepath"] = inv_row["filepath"]
    ann["image_width"] = w
    ann["image_height"] = h
    ann["review_status"] = state["review_status"]
    ann["shadows_present"] = state["shadows_present"]
    ann["patches_present"] = state["patches_present"]
    ann["manholes_present"] = state["manholes_present"]
    ann["uncertain"] = state["uncertain"]
    ann["notes"] = state["notes"]
    ann["pothole_boxes"] = state.get("pothole_boxes", [])
    ann["negative_regions"] = state.get("negative_regions", [])
    saved_path = ann_io.save(ann)

    inv_updates = {
        "review_status": state["review_status"],
        "pothole_count": str(len(state.get("pothole_boxes", []))),
        "shadows_present": str(state["shadows_present"]).lower(),
        "patches_present": str(state["patches_present"]).lower(),
        "manholes_present": str(state["manholes_present"]).lower(),
        "uncertain": str(state["uncertain"]).lower(),
        "notes": state["notes"],
    }
    update_inventory_record(image_id, inv_updates)
    inv_row.update(inv_updates)

    st.session_state[f"last_saved_{image_id}"] = {
        "time": datetime.now().strftime("%I:%M:%S %p"),
        "path": rel(saved_path),
        "status": state["review_status"],
        "potholes": len(state.get("pothole_boxes", [])),
        "negatives": len(state.get("negative_regions", [])),
    }
    st.cache_data.clear()
    return saved_path


def get_box_snapshot(state: dict) -> dict:
    return {
        "pothole_boxes": [dict(b) for b in state.get("pothole_boxes", [])],
        "negative_regions": [dict(nr) for nr in state.get("negative_regions", [])],
    }


def init_history(image_id: str, state: dict) -> None:
    h_key = f"box_history_{image_id}"
    idx_key = f"box_history_idx_{image_id}"
    if h_key not in st.session_state:
        st.session_state[h_key] = [get_box_snapshot(state)]
        st.session_state[idx_key] = 0


def push_history(image_id: str, state: dict) -> None:
    h_key = f"box_history_{image_id}"
    idx_key = f"box_history_idx_{image_id}"
    init_history(image_id, state)
    cur_idx = st.session_state[idx_key]
    history = st.session_state[h_key][: cur_idx + 1]
    new_snap = get_box_snapshot(state)
    if history and history[-1] == new_snap:
        return
    history.append(new_snap)
    st.session_state[h_key] = history
    st.session_state[idx_key] = len(history) - 1


def can_undo(image_id: str) -> bool:
    idx_key = f"box_history_idx_{image_id}"
    return st.session_state.get(idx_key, 0) > 0


def can_redo(image_id: str) -> bool:
    h_key = f"box_history_{image_id}"
    idx_key = f"box_history_idx_{image_id}"
    history = st.session_state.get(h_key, [])
    idx = st.session_state.get(idx_key, 0)
    return idx < len(history) - 1


def do_undo(image_id: str, inv_row: dict, state: dict) -> None:
    if not can_undo(image_id):
        return
    idx_key = f"box_history_idx_{image_id}"
    h_key = f"box_history_{image_id}"
    st.session_state[idx_key] -= 1
    snap = st.session_state[h_key][st.session_state[idx_key]]
    state["pothole_boxes"] = [dict(b) for b in snap["pothole_boxes"]]
    state["negative_regions"] = [dict(nr) for nr in snap["negative_regions"]]
    st.session_state[f"canvas_ver_{image_id}"] += 1
    save_annotation_from_state(image_id, inv_row, state)
    st.rerun()


def do_redo(image_id: str, inv_row: dict, state: dict) -> None:
    if not can_redo(image_id):
        return
    idx_key = f"box_history_idx_{image_id}"
    h_key = f"box_history_{image_id}"
    st.session_state[idx_key] += 1
    snap = st.session_state[h_key][st.session_state[idx_key]]
    state["pothole_boxes"] = [dict(b) for b in snap["pothole_boxes"]]
    state["negative_regions"] = [dict(nr) for nr in snap["negative_regions"]]
    st.session_state[f"canvas_ver_{image_id}"] += 1
    save_annotation_from_state(image_id, inv_row, state)
    st.rerun()


def parse_fabric_rect(
    obj: dict, scale_x: float, scale_y: float, img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
    """Accurately extract top-left image coordinates from a Fabric.js rectangle object.

    Normalizes originX and originY (center/right/bottom), scaleX, scaleY, and negative
    drag directions, preventing any coordinate shifting.
    """
    ox = str(obj.get("originX", "left")).lower()
    oy = str(obj.get("originY", "top")).lower()
    sx = float(obj.get("scaleX", 1.0))
    sy = float(obj.get("scaleY", 1.0))

    bw = float(obj.get("width", 0.0)) * sx
    bh = float(obj.get("height", 0.0)) * sy
    bl = float(obj.get("left", 0.0))
    bt = float(obj.get("top", 0.0))

    # Normalize origin to top-left
    if ox == "center":
        bl -= abs(bw) / 2.0
    elif ox == "right":
        bl -= abs(bw)

    if oy == "center":
        bt -= abs(bh) / 2.0
    elif oy == "bottom":
        bt -= abs(bh)

    # Normalize inverted dragging (e.g. drawn from bottom-right to top-left)
    if bw < 0:
        bl += bw
        bw = abs(bw)
    if bh < 0:
        bt += bh
        bh = abs(bh)

    if bw < 4 or bh < 4:
        return None

    x = max(0, min(img_w - 5, int(round(bl / scale_x))))
    y = max(0, min(img_h - 5, int(round(bt / scale_y))))
    w = max(5, min(img_w - x, int(round(bw / scale_x))))
    h = max(5, min(img_h - y, int(round(bh / scale_y))))
    return x, y, w, h


def render_annotated_image(pil_img: Image.Image, boxes: list[dict], neg_regions: list[dict]) -> Image.Image:
    img_copy = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)

    # Red boxes for Potholes
    for idx, b in enumerate(boxes):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline="#dc2626", width=4)
        tag = f"Pothole #{idx + 1}"
        draw.rectangle([x, max(0, y - 22), x + len(tag) * 9 + 8, y], fill="#dc2626")
        draw.text((x + 4, max(0, y - 20)), tag, fill="#ffffff")

    # Color-coded Negative regions
    for idx, nr in enumerate(neg_regions):
        x, y, w, h = nr["x"], nr["y"], nr["w"], nr["h"]
        x2, y2 = x + w, y + h
        subtype = nr.get("subtype", "repaired_patch")
        col = {
            "shadow": "#9333ea",
            "manhole": "#059669",
        }.get(subtype, "#2563eb")

        draw.rectangle([x, y, x2, y2], outline=col, width=4)
        tag = f"{subtype} #{idx + 1}"
        draw.rectangle([x, max(0, y - 22), x + len(tag) * 9 + 8, y], fill=col)
        draw.text((x + 4, max(0, y - 20)), tag, fill="#ffffff")

    return img_copy




# ---------------------------------------------------------------------------
# Load Inventory Data
# ---------------------------------------------------------------------------
inventory = load_inventory()
if not inventory:
    st.error(
        "Image inventory not found. Generate inventory first:\n\n"
        "python scripts/build_inventory.py"
    )
    st.stop()

video_frame_index = load_video_frame_index()

# ---------------------------------------------------------------------------
# Sidebar: Navigation, Filters, Native Metrics, and Batch Tools
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Navigation & Filter")

    source_options = sorted({r.get("source", "unknown") for r in inventory})
    status_options = ["all"] + sorted({r.get("review_status", "unreviewed") for r in inventory})

    filter_source = st.selectbox("Source Filter", ["all"] + source_options, key="sb_src_filter")
    filter_status = st.selectbox("Status Filter", status_options, key="sb_stat_filter")
    filter_sequence = st.text_input("Filter Sequence / Video ID", "", key="sb_seq_filter")

    filtered = [
        r for r in inventory
        if (filter_source == "all" or r.get("source") == filter_source)
        and (filter_status == "all" or r.get("review_status") == filter_status)
        and (not filter_sequence or filter_sequence in r.get("sequence_id", "") + r.get("recording_id", ""))
    ]

    st.caption(f"Showing {len(filtered)} of {len(inventory)} images")

    if not filtered:
        st.warning("No records match the active criteria.")
        st.stop()

    def label_for(r: dict) -> str:
        status_tag = r.get("review_status", "unreviewed").upper()
        return f"[{status_tag}] {r['original_filename']}"

    options = [label_for(r) for r in filtered]
    image_ids = [r["image_id"] for r in filtered]

    if "selected_image_id" not in st.session_state or st.session_state.selected_image_id not in image_ids:
        st.session_state.selected_image_id = image_ids[0]

    current_idx = image_ids.index(st.session_state.selected_image_id)

    def save_current_image_if_present() -> None:
        curr_id = st.session_state.get("selected_image_id")
        if not curr_id:
            return
        sk = f"ann_state_{curr_id}"
        if sk in st.session_state:
            curr_state = st.session_state[sk]
            matching = [r for r in inventory if r.get("image_id") == curr_id]
            if matching:
                save_annotation_from_state(curr_id, matching[0], curr_state)

    def on_nav_first() -> None:
        save_current_image_if_present()
        st.session_state.selected_image_id = image_ids[0]

    def on_nav_prev() -> None:
        save_current_image_if_present()
        idx = image_ids.index(st.session_state.selected_image_id)
        if idx > 0:
            st.session_state.selected_image_id = image_ids[idx - 1]

    def on_nav_next() -> None:
        save_current_image_if_present()
        idx = image_ids.index(st.session_state.selected_image_id)
        if idx < len(image_ids) - 1:
            st.session_state.selected_image_id = image_ids[idx + 1]

    def on_dropdown_change() -> None:
        save_current_image_if_present()
        selected_text = st.session_state.image_selector_dropdown
        try:
            idx = options.index(selected_text)
            st.session_state.selected_image_id = image_ids[idx]
        except (ValueError, IndexError):
            pass

    col_nav1, col_nav2, col_nav3 = st.columns([1.1, 1.3, 1.1])
    with col_nav1:
        st.button(
            "← Prev",
            use_container_width=True,
            on_click=on_nav_prev,
            disabled=(current_idx <= 0),
            key="sb_btn_prev",
        )
    with col_nav2:
        st.button(
            "⏮ Start (1)",
            use_container_width=True,
            on_click=on_nav_first,
            disabled=(current_idx <= 0),
            help="Jump back to the first image (#1)",
            key="sb_btn_first",
        )
    with col_nav3:
        st.button(
            "Next →",
            use_container_width=True,
            on_click=on_nav_next,
            disabled=(current_idx >= len(filtered) - 1),
            key="sb_btn_next",
        )

    st.selectbox(
        "Select Image",
        options,
        index=current_idx,
        key="image_selector_dropdown",
        on_change=on_dropdown_change,
    )

    auto_advance = st.checkbox("Auto-advance to next image after review", value=True)

    st.divider()

    # Progress Metrics using Native Streamlit st.metric (Prevents white-on-white text)
    reviewed_count = sum(1 for r in inventory if r.get("review_status") == "reviewed")
    uncertain_count = sum(1 for r in inventory if r.get("review_status") == "uncertain")
    unreviewed_count = len(inventory) - (reviewed_count + uncertain_count)
    pct_complete = (reviewed_count / len(inventory)) * 100 if inventory else 0.0

    st.markdown("### Dataset Progress")
    st.progress(pct_complete / 100.0)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Total Images", len(inventory))
        st.metric("Pending", unreviewed_count)
    with m_col2:
        st.metric("Reviewed", f"{reviewed_count} ({pct_complete:.1f}%)")
        st.metric("Uncertain", uncertain_count)

    st.divider()

    # Batch Tools
    with st.expander("Batch Review Tools"):
        st.caption("Apply review operations across all images.")
        if st.button("Mark All Unreviewed as Repaired Patches", use_container_width=True):
            count_marked = 0
            for r in inventory:
                if r.get("review_status") != "reviewed":
                    img_id = r["image_id"]
                    batch_state = {
                        "review_status": "reviewed",
                        "shadows_present": False,
                        "patches_present": True,
                        "manholes_present": False,
                        "uncertain": False,
                        "notes": "Verified repaired road surface (no open potholes).",
                        "pothole_boxes": [],
                        "negative_regions": [],
                    }
                    save_annotation_from_state(img_id, r, batch_state)
                    count_marked += 1
            st.success(f"Marked {count_marked} images as reviewed repaired patches.")
            st.cache_data.clear()
            st.rerun()

        if st.button("Auto-Add Center Patch Crop to All Repaired", use_container_width=True):
            count_crops = 0
            for r in inventory:
                if r.get("review_status") == "reviewed" and r.get("patches_present") == "true":
                    img_id = r["image_id"]
                    cur_ann = ann_io.load(img_id)
                    w = int(r.get("width") or 2048)
                    h = int(r.get("height") or 1536)
                    if not cur_ann.get("negative_regions"):
                        cur_ann["negative_regions"] = [
                            {
                                "x": int(w * 0.30),
                                "y": int(h * 0.60),
                                "w": int(w * 0.40),
                                "h": int(h * 0.25),
                                "subtype": "repaired_patch",
                            }
                        ]
                        ann_io.save(cur_ann)
                        count_crops += 1
            st.success(f"Added default center patch region to {count_crops} images.")
            st.cache_data.clear()
            st.rerun()

inv_row = filtered[current_idx]
image_id = inv_row["image_id"]
img_path = PROJECT_ROOT / inv_row["filepath"]
frame_meta = video_frame_index.get(inv_row["filepath"], {})

# ---------------------------------------------------------------------------
# Annotation State Management
# ---------------------------------------------------------------------------
ann = ann_io.load(image_id)
state_key = f"ann_state_{image_id}"

if state_key not in st.session_state:
    st.session_state[state_key] = {
        "review_status": ann.get("review_status", inv_row.get("review_status", "unreviewed")),
        "shadows_present": ann.get("shadows_present", inv_row.get("shadows_present") == "true"),
        "patches_present": ann.get("patches_present", inv_row.get("patches_present") == "true"),
        "manholes_present": ann.get("manholes_present", inv_row.get("manholes_present") == "true"),
        "uncertain": ann.get("uncertain", inv_row.get("uncertain") == "true"),
        "notes": ann.get("notes", inv_row.get("notes", "")),
        "pothole_boxes": ann.get("pothole_boxes", []),
        "negative_regions": ann.get("negative_regions", []),
    }

state = st.session_state[state_key]
init_history(image_id, state)

# Track disk file status
ann_disk_path = ann_io.annotation_path(image_id)
if f"last_saved_{image_id}" not in st.session_state:
    if ann_disk_path.exists():
        mtime = datetime.fromtimestamp(ann_disk_path.stat().st_mtime).strftime("%I:%M:%S %p")
        st.session_state[f"last_saved_{image_id}"] = {
            "time": mtime,
            "path": rel(ann_disk_path),
            "status": ann.get("review_status", inv_row.get("review_status", "unreviewed")),
            "potholes": len(ann.get("pothole_boxes", [])),
            "negatives": len(ann.get("negative_regions", [])),
        }
    else:
        st.session_state[f"last_saved_{image_id}"] = {
            "time": "Not yet saved to disk",
            "path": rel(ann_disk_path),
            "status": state["review_status"],
            "potholes": len(state.get("pothole_boxes", [])),
            "negatives": len(state.get("negative_regions", [])),
        }

# Version counter to reset canvas only when explicitly triggered
canvas_ver_key = f"canvas_ver_{image_id}"
if canvas_ver_key not in st.session_state:
    st.session_state[canvas_ver_key] = 0

# ---------------------------------------------------------------------------
# Clean Top Navigation & Status Bar
# ---------------------------------------------------------------------------
status_style = {
    "reviewed": "status-reviewed",
    "uncertain": "status-uncertain",
}.get(state["review_status"], "status-unreviewed")

ts_text = ""
if frame_meta.get("timestamp_sec"):
    try:
        ts = float(frame_meta["timestamp_sec"])
        ts_text = f" • {ts:.1f}s"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Workspace Dispatch State
# ---------------------------------------------------------------------------
is_fullscreen_active = st.session_state.get("fullscreen_mode", False)

# Top control bar: Navigation + Header Meta + Quick Save (Hidden in Fullscreen to prevent duplicate bars)
if not is_fullscreen_active:
    top_c1, top_c2, top_c3 = st.columns([1.2, 3.5, 1.5], vertical_alignment="center")
    with top_c1:
        h_nav1, h_nav2 = st.columns(2)
        with h_nav1:
            st.button("← Prev", use_container_width=True, on_click=on_nav_prev, disabled=(current_idx <= 0), key="h_btn_prev")
        with h_nav2:
            st.button("Next →", use_container_width=True, on_click=on_nav_next, disabled=(current_idx >= len(filtered) - 1), key="h_btn_next")

    with top_c2:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; overflow: hidden; white-space: nowrap;">
                <strong style="font-size: 1.05rem;">{inv_row['original_filename']}</strong>
                <span class="status-pill {status_style}">{state['review_status']}</span>
                <span style="font-size: 0.8rem; opacity: 0.7;">
                    ({current_idx + 1} of {len(filtered)}) • {inv_row.get('source', '-')}{ts_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_c3:
        if st.button("Save & Next →", type="primary", use_container_width=True, key="h_btn_save_next"):
            state["review_status"] = "reviewed"
            state["uncertain"] = False
            save_annotation_from_state(image_id, inv_row, state)
            if current_idx < len(filtered) - 1:
                st.session_state.selected_image_id = image_ids[current_idx + 1]
            st.rerun()

    st.divider()

# ---------------------------------------------------------------------------
# Load PIL Image
# ---------------------------------------------------------------------------
pil_img = None
img_w, img_h = 2048, 1536
if img_path.exists():
    try:
        pil_img = Image.open(img_path)
        img_w, img_h = pil_img.size
    except Exception as exc:
        st.error(f"Cannot read image: {exc}")


# ---------------------------------------------------------------------------
# Canvas Studio (Unified Component for Standard & Fullscreen Modes)
# ---------------------------------------------------------------------------
def render_canvas_studio(is_fullscreen: bool = False) -> None:
    # Inject keyboard shortcuts listener into parent document
    shortcuts_js = """
    <script>
    (function() {
        const parentDoc = window.parent.document;
        if (!parentDoc || parentDoc._pothole_shortcuts_active) return;
        parentDoc._pothole_shortcuts_active = true;

        parentDoc.addEventListener('keydown', function(e) {
            const active = parentDoc.activeElement;
            const tag = active ? active.tagName.toLowerCase() : '';
            if (tag === 'input' || tag === 'textarea' || (active && active.isContentEditable)) {
                return;
            }

            const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
            const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

            // 1. Undo: Cmd+Z or Ctrl+Z (without Shift)
            if (cmdOrCtrl && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && b.innerText.includes('Undo') && !b.disabled) {
                        b.click();
                        return;
                    }
                }
            }

            // 2. Redo: Cmd+Shift+Z, Ctrl+Shift+Z, or Cmd+Y, Ctrl+Y
            if (
                (cmdOrCtrl && (e.key === 'z' || e.key === 'Z') && e.shiftKey) ||
                (cmdOrCtrl && (e.key === 'y' || e.key === 'Y'))
            ) {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && b.innerText.includes('Redo') && !b.disabled) {
                        b.click();
                        return;
                    }
                }
            }

            // 3. Zoom In: '+' or '=' without modifier
            if (!cmdOrCtrl && !e.altKey && (e.key === '+' || e.key === '=')) {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && b.innerText.trim() === '+' && !b.disabled) {
                        b.click();
                        return;
                    }
                }
            }

            // 4. Zoom Out: '-' or '_' without modifier
            if (!cmdOrCtrl && !e.altKey && (e.key === '-' || e.key === '_')) {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && (b.innerText.trim() === '−' || b.innerText.trim() === '-') && !b.disabled) {
                        b.click();
                        return;
                    }
                }
            }

            // 5. Reset Zoom: '0' without modifier
            if (!cmdOrCtrl && !e.altKey && e.key === '0') {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && b.innerText.trim().endsWith('%')) {
                        b.click();
                        return;
                    }
                }
            }

            // 6. Delete Last Box: Backspace or Delete without modifier
            if (!cmdOrCtrl && !e.altKey && (e.key === 'Backspace' || e.key === 'Delete')) {
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && (b.innerText.includes('Del Last') || b.innerText.includes('Del')) && !b.disabled) {
                        e.preventDefault();
                        e.stopPropagation();
                        b.click();
                        return;
                    }
                }
            }

            // 7. Quick Class Switch: 1, 2, 3, 4
            if (!cmdOrCtrl && !e.altKey && ['1', '2', '3', '4'].includes(e.key)) {
                const idx = parseInt(e.key, 10) - 1;
                const radios = parentDoc.querySelectorAll('div[data-testid="stRadio"] input[type="radio"]');
                if (radios && radios[idx]) {
                    e.preventDefault();
                    e.stopPropagation();
                    radios[idx].click();
                    return;
                }
            }

            // 8. Toggle Fullscreen Mode: 'f' or 'F' without modifier
            if (!cmdOrCtrl && !e.altKey && (e.key === 'f' || e.key === 'F')) {
                e.preventDefault();
                e.stopPropagation();
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && (b.innerText.includes('Full') || b.innerText.includes('Standard View'))) {
                        b.click();
                        return;
                    }
                }
            }

            // 9. Exit Fullscreen on Escape
            if (e.key === 'Escape') {
                for (const b of parentDoc.querySelectorAll('button')) {
                    if (b.innerText && b.innerText.includes('Standard View')) {
                        e.preventDefault();
                        e.stopPropagation();
                        b.click();
                        return;
                    }
                }
            }
        }, true);
    })();
    </script>
    """
    components.html(shortcuts_js, height=0, width=0)

    # Zoom factor tracking
    ZOOM_LEVELS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    if "global_zoom_mult" not in st.session_state:
        st.session_state["global_zoom_mult"] = 1.0
    current_zoom = float(st.session_state["global_zoom_mult"])
    zoom_idx = min(range(len(ZOOM_LEVELS)), key=lambda i: abs(ZOOM_LEVELS[i] - current_zoom))

    num_p = len(state["pothole_boxes"])
    num_n = len(state["negative_regions"])
    total_boxes = num_p + num_n

    if is_fullscreen:
        # Fullscreen Top Control Bar: Exit + Nav + Review Actions + Save & Next
        fs_bar1, fs_bar2, fs_bar3, fs_bar4 = st.columns([1.6, 2.8, 3.2, 1.4], vertical_alignment="center")
        with fs_bar1:
            if st.button("← Standard View", key=f"fs_exit_top_{image_id}", use_container_width=True, help="Exit Fullscreen Studio (Key: Esc or F)"):
                st.session_state["fullscreen_mode"] = False
                st.rerun()
        with fs_bar2:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 0.6rem; overflow: hidden; white-space: nowrap;">
                    <strong style="font-size: 1.1rem;">{inv_row['original_filename']}</strong>
                    <span class="status-pill {status_style}">{state['review_status']}</span>
                    <span style="font-size: 0.85rem; opacity: 0.75;">({current_idx + 1}/{len(filtered)})</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with fs_bar3:
            qc1, qc2, qc3 = st.columns(3)
            with qc1:
                if st.button("Clean (0)", key=f"fs_btn_clean_{image_id}", use_container_width=True, help="Clean Road - 0 Potholes"):
                    state["review_status"] = "reviewed"
                    state["patches_present"] = False
                    state["pothole_boxes"] = []
                    state["uncertain"] = False
                    save_annotation_from_state(image_id, inv_row, state)
                    if auto_advance and current_idx < len(filtered) - 1:
                        st.session_state.selected_image_id = image_ids[current_idx + 1]
                    st.rerun()
            with qc2:
                if st.button("Repaired (0)", key=f"fs_btn_rep_{image_id}", use_container_width=True, help="Repaired Road - 0 Potholes"):
                    state["review_status"] = "reviewed"
                    state["patches_present"] = True
                    state["pothole_boxes"] = []
                    state["uncertain"] = False
                    save_annotation_from_state(image_id, inv_row, state)
                    if auto_advance and current_idx < len(filtered) - 1:
                        st.session_state.selected_image_id = image_ids[current_idx + 1]
                    st.rerun()
            with qc3:
                if st.button("Uncertain", key=f"fs_btn_unc_{image_id}", use_container_width=True):
                    state["review_status"] = "uncertain"
                    state["uncertain"] = True
                    save_annotation_from_state(image_id, inv_row, state)
                    if auto_advance and current_idx < len(filtered) - 1:
                        st.session_state.selected_image_id = image_ids[current_idx + 1]
                    st.rerun()
        with fs_bar4:
            if st.button("Save & Next →", type="primary", key=f"fs_btn_save_{image_id}", use_container_width=True):
                state["review_status"] = "reviewed"
                state["uncertain"] = False
                save_annotation_from_state(image_id, inv_row, state)
                if current_idx < len(filtered) - 1:
                    st.session_state.selected_image_id = image_ids[current_idx + 1]
                st.rerun()

        st.divider()

        # Fullscreen Studio Toolbar: Class Selection (Left) + Editing/Zoom Tools (Right)
        tb_col_class, tb_col_actions = st.columns([1.6, 2.4], vertical_alignment="center")
        with tb_col_class:
            active_class = st.radio(
                "Class",
                [
                    "1 Pothole (Red)",
                    "2 Patch (Blue)",
                    "3 Shadow (Purple)",
                    "4 Manhole (Green)",
                ],
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                key=f"fs_class_{image_id}",
            )
        with tb_col_actions:
            e1, e2, e3, e4, sp, z1, z2, z3 = st.columns([1.2, 1.2, 1.3, 1.2, 0.3, 0.8, 1.2, 0.8])
            with e1:
                if st.button("↶ Undo", key=f"fs_undo_{image_id}", disabled=not can_undo(image_id), use_container_width=True, help="Undo last box drawn or modified (Ctrl+Z)"):
                    do_undo(image_id, inv_row, state)
            with e2:
                if st.button("↷ Redo", key=f"fs_redo_{image_id}", disabled=not can_redo(image_id), use_container_width=True, help="Redo previously undone action (Ctrl+Y)"):
                    do_redo(image_id, inv_row, state)
            with e3:
                if st.button("⌫ Del Last", key=f"fs_dellast_{image_id}", disabled=(total_boxes == 0), use_container_width=True, help="Delete the most recently created box (Backspace)"):
                    if state["negative_regions"]:
                        state["negative_regions"].pop()
                    elif state["pothole_boxes"]:
                        state["pothole_boxes"].pop()
                    push_history(image_id, state)
                    st.session_state[canvas_ver_key] += 1
                    save_annotation_from_state(image_id, inv_row, state)
                    st.rerun()
            with e4:
                if st.button("✕ Clear", key=f"fs_clear_{image_id}", disabled=(total_boxes == 0), use_container_width=True, help="Clear all boxes on this image (Undoable with Ctrl+Z)"):
                    state["pothole_boxes"] = []
                    state["negative_regions"] = []
                    push_history(image_id, state)
                    st.session_state[canvas_ver_key] += 1
                    save_annotation_from_state(image_id, inv_row, state)
                    st.rerun()
            with sp:
                st.write("")
            with z1:
                if st.button("−", key=f"fs_zm_out_{image_id}", disabled=(zoom_idx <= 0), use_container_width=True, help="Zoom Out (- key)"):
                    st.session_state["global_zoom_mult"] = ZOOM_LEVELS[zoom_idx - 1]
                    st.rerun()
            with z2:
                if st.button(f"{int(current_zoom * 100)}%", key=f"fs_zm_rst_{image_id}", use_container_width=True, help="Reset Zoom to 100% normal view (0 key)"):
                    st.session_state["global_zoom_mult"] = 1.0
                    st.rerun()
            with z3:
                if st.button("+", key=f"fs_zm_in_{image_id}", disabled=(zoom_idx >= len(ZOOM_LEVELS) - 1), use_container_width=True, help="Zoom In (+ key)"):
                    st.session_state["global_zoom_mult"] = ZOOM_LEVELS[zoom_idx + 1]
                    st.rerun()

        # Spacious canvas dimensions for Fullscreen
        base_h = 680
        container_h = 720

    else:
        # Standard View: Clean 2-Row Toolbar with generous spacing
        active_class = st.radio(
            "Class",
            [
                "1 Pothole (Red)",
                "2 Patch (Blue)",
                "3 Shadow (Purple)",
                "4 Manhole (Green)",
            ],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
            key=f"std_class_{image_id}",
        )

        col_tb_edit, col_tb_zoom = st.columns([1.1, 1.0], vertical_alignment="center")
        with col_tb_edit:
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                if st.button("↶ Undo", key=f"btn_undo_{image_id}", disabled=not can_undo(image_id), use_container_width=True, help="Undo last box drawn or modified (Ctrl+Z)"):
                    do_undo(image_id, inv_row, state)
            with e2:
                if st.button("↷ Redo", key=f"btn_redo_{image_id}", disabled=not can_redo(image_id), use_container_width=True, help="Redo previously undone action (Ctrl+Y)"):
                    do_redo(image_id, inv_row, state)
            with e3:
                if st.button("⌫ Del", key=f"btn_del_last_{image_id}", disabled=(total_boxes == 0), use_container_width=True, help="Delete the most recently created box (Backspace)"):
                    if state["negative_regions"]:
                        state["negative_regions"].pop()
                    elif state["pothole_boxes"]:
                        state["pothole_boxes"].pop()
                    push_history(image_id, state)
                    st.session_state[canvas_ver_key] += 1
                    save_annotation_from_state(image_id, inv_row, state)
                    st.rerun()
            with e4:
                if st.button("✕ Clear", key=f"btn_clear_all_{image_id}", disabled=(total_boxes == 0), use_container_width=True, help="Clear all boxes on this image (Undoable with Ctrl+Z)"):
                    state["pothole_boxes"] = []
                    state["negative_regions"] = []
                    push_history(image_id, state)
                    st.session_state[canvas_ver_key] += 1
                    save_annotation_from_state(image_id, inv_row, state)
                    st.rerun()

        with col_tb_zoom:
            z1, z2, z3, z4 = st.columns([0.8, 1.3, 0.8, 1.7])
            with z1:
                if st.button("−", key=f"btn_zoom_out_{image_id}", disabled=(zoom_idx <= 0), use_container_width=True, help="Zoom Out (- key)"):
                    st.session_state["global_zoom_mult"] = ZOOM_LEVELS[zoom_idx - 1]
                    st.rerun()
            with z2:
                if st.button(f"{int(current_zoom * 100)}%", key=f"btn_zoom_reset_{image_id}", use_container_width=True, help="Reset Zoom to 100% normal view (0 key)"):
                    st.session_state["global_zoom_mult"] = 1.0
                    st.rerun()
            with z3:
                if st.button("+", key=f"btn_zoom_in_{image_id}", disabled=(zoom_idx >= len(ZOOM_LEVELS) - 1), use_container_width=True, help="Zoom In (+ key)"):
                    st.session_state["global_zoom_mult"] = ZOOM_LEVELS[zoom_idx + 1]
                    st.rerun()
            with z4:
                if st.button("⛶ Full", key=f"btn_full_{image_id}", use_container_width=True, help="Open Fullscreen Canvas Studio (F key)"):
                    st.session_state["fullscreen_mode"] = True
                    st.rerun()

        # Canvas Dimensions for Standard View: base_h = 460 ensures disp_w = 613px, comfortably fitting inside 680px column!
        base_h = 460
        container_h = 500

    disp_h = int(base_h * current_zoom)
    scale_y = disp_h / float(img_h)
    disp_w = int(img_w * scale_y)
    scale_x = disp_w / float(img_w)

    # Active drawing styling
    c_mode = "rect"
    if "Pothole" in active_class:
        active_stroke = "#dc2626"
        active_fill = "rgba(220, 38, 38, 0.25)"
    elif "Shadow" in active_class:
        active_stroke = "#9333ea"
        active_fill = "rgba(147, 51, 234, 0.25)"
    elif "Manhole" in active_class:
        active_stroke = "#059669"
        active_fill = "rgba(5, 150, 105, 0.25)"
    else:
        active_stroke = "#2563eb"
        active_fill = "rgba(37, 99, 235, 0.25)"

    # Build initial objects with explicit origins and uniform stroke
    initial_objects = []
    for b in state["pothole_boxes"]:
        initial_objects.append({
            "type": "rect",
            "left": int(round(b["x"] * scale_x)),
            "top": int(round(b["y"] * scale_y)),
            "width": int(round(b["w"] * scale_x)),
            "height": int(round(b["h"] * scale_y)),
            "originX": "left",
            "originY": "top",
            "scaleX": 1.0,
            "scaleY": 1.0,
            "angle": 0,
            "fill": "rgba(220, 38, 38, 0.25)",
            "stroke": "#dc2626",
            "strokeWidth": 2,
            "strokeUniform": True,
            "lockRotation": True,
        })
    for nr in state["negative_regions"]:
        stype = nr.get("subtype", "repaired_patch")
        c_hex, f_rgba = {
            "shadow": ("#9333ea", "rgba(147, 51, 234, 0.25)"),
            "manhole": ("#059669", "rgba(5, 150, 105, 0.25)"),
        }.get(stype, ("#2563eb", "rgba(37, 99, 235, 0.25)"))
        initial_objects.append({
            "type": "rect",
            "left": int(round(nr["x"] * scale_x)),
            "top": int(round(nr["y"] * scale_y)),
            "width": int(round(nr["w"] * scale_x)),
            "height": int(round(nr["h"] * scale_y)),
            "originX": "left",
            "originY": "top",
            "scaleX": 1.0,
            "scaleY": 1.0,
            "angle": 0,
            "fill": f_rgba,
            "stroke": c_hex,
            "strokeWidth": 2,
            "strokeUniform": True,
            "lockRotation": True,
        })

    # Clear visual buffer ensuring buttons are separated from container border
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Fixed-height scrollable viewport container
    with st.container(height=container_h):
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        canvas_key = f"canvas_{image_id}_{st.session_state[canvas_ver_key]}_{'fs' if is_fullscreen else 'std'}"
        if CANVAS_AVAILABLE and pil_img is not None:
            canvas_result = st_canvas(
                fill_color=active_fill,
                stroke_width=2,
                stroke_color=active_stroke,
                background_image=pil_img.resize((disp_w, disp_h), Image.Resampling.LANCZOS),
                update_streamlit=True,
                width=disp_w,
                height=disp_h,
                drawing_mode=c_mode,
                initial_drawing={"version": "4.4.0", "objects": initial_objects} if initial_objects else None,
                max_display_height=container_h - 24,
                key=canvas_key,
            )

            existing_box_count = len(state["pothole_boxes"]) + len(state["negative_regions"])
            if canvas_result.json_data is not None:
                raw_objects = canvas_result.json_data.get("objects", [])
                rect_objects = [
                    obj for obj in raw_objects
                    if str(obj.get("type", "")).lower() in ("rect", "labeledrect")
                ]

                if len(rect_objects) > existing_box_count:
                    new_items = rect_objects[existing_box_count:]
                    added = False
                    for obj in new_items:
                        parsed = parse_fabric_rect(obj, scale_x, scale_y, img_w, img_h)
                        if parsed is None:
                            continue
                        bx, by, bw, bh = parsed

                        if "Pothole" in active_class:
                            state["pothole_boxes"].append({"x": bx, "y": by, "w": bw, "h": bh})
                        elif "Shadow" in active_class:
                            state["negative_regions"].append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "shadow"})
                            state["shadows_present"] = True
                        elif "Manhole" in active_class:
                            state["negative_regions"].append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "manhole"})
                            state["manholes_present"] = True
                        else:
                            state["negative_regions"].append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "repaired_patch"})
                            state["patches_present"] = True
                        added = True

                    if added:
                        state["review_status"] = "reviewed"
                        push_history(image_id, state)
                        st.session_state[canvas_ver_key] += 1
                        save_annotation_from_state(image_id, inv_row, state)
                        st.rerun()

                elif len(rect_objects) == existing_box_count and existing_box_count > 0:
                    modified = False
                    p_len = len(state["pothole_boxes"])
                    for idx, obj in enumerate(rect_objects):
                        parsed = parse_fabric_rect(obj, scale_x, scale_y, img_w, img_h)
                        if parsed is None:
                            continue
                        nx, ny, nw, nh = parsed
                        if idx < p_len:
                            cur = state["pothole_boxes"][idx]
                        else:
                            cur = state["negative_regions"][idx - p_len]
                        if (
                            abs(nx - cur["x"]) > 2
                            or abs(ny - cur["y"]) > 2
                            or abs(nw - cur["w"]) > 2
                            or abs(nh - cur["h"]) > 2
                        ):
                            cur["x"], cur["y"], cur["w"], cur["h"] = nx, ny, nw, nh
                            modified = True

                    if modified:
                        push_history(image_id, state)
                        st.session_state[canvas_ver_key] += 1
                        save_annotation_from_state(image_id, inv_row, state)
                        st.rerun()

                elif len(rect_objects) < existing_box_count:
                    # Sync any deleted boxes from canvas directly back to state & disk
                    new_potholes = []
                    new_negatives = []
                    for obj in rect_objects:
                        parsed = parse_fabric_rect(obj, scale_x, scale_y, img_w, img_h)
                        if parsed is None:
                            continue
                        bx, by, bw, bh = parsed
                        s_str = str(obj.get("stroke", "")).lower()
                        f_str = str(obj.get("fill", "")).lower()
                        if "#dc2626" in s_str or "220, 38" in f_str:
                            new_potholes.append({"x": bx, "y": by, "w": bw, "h": bh})
                        elif "#9333ea" in s_str or "147, 51" in f_str:
                            new_negatives.append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "shadow"})
                        elif "#059669" in s_str or "5, 150" in f_str:
                            new_negatives.append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "manhole"})
                        else:
                            new_negatives.append({"x": bx, "y": by, "w": bw, "h": bh, "subtype": "repaired_patch"})

                    state["pothole_boxes"] = new_potholes
                    state["negative_regions"] = new_negatives
                    push_history(image_id, state)
                    st.session_state[canvas_ver_key] += 1
                    save_annotation_from_state(image_id, inv_row, state)
                    st.rerun()

        else:
            if pil_img is not None:
                rendered = render_annotated_image(pil_img, state["pothole_boxes"], state["negative_regions"])
                st.image(rendered.resize((disp_w, disp_h)), use_container_width=False)

    # Box Chip Bar
    if total_boxes == 0:
        st.caption("💡 Shortcuts: 1-4 Class | Ctrl+Z Undo | Ctrl+Y Redo | Backspace Del Last | +/- Zoom | 0 Reset | F Fullscreen")
    else:
        chips = []
        if num_p > 0:
            chips.append(f"<span style='color: #dc2626; font-weight: 600;'>{num_p} Pothole{'s' if num_p > 1 else ''}</span>")
        if num_n > 0:
            chips.append(f"<span style='color: #2563eb; font-weight: 600;'>{num_n} Negative Region{'s' if num_n > 1 else ''}</span>")
        st.markdown(" • ".join(chips), unsafe_allow_html=True)

    # Detailed Coordinate Inspector (Collapsed by default for clean UI)
    if total_boxes > 0:
        with st.expander(f"Box Details & Coordinate Adjuster ({total_boxes})", expanded=False):
            changed_manual = False
            for i, box in enumerate(list(state["pothole_boxes"])):
                c_lbl, c_x, c_y, c_w, c_h, c_del = st.columns([1.5, 1, 1, 1, 1, 0.8])
                with c_lbl:
                    st.markdown(f"<span style='color:#dc2626; font-weight:600;'>Pothole #{i+1}</span>", unsafe_allow_html=True)
                with c_x:
                    nx = st.number_input("X", 0, img_w, box["x"], 10, key=f"x_p_{i}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with c_y:
                    ny = st.number_input("Y", 0, img_h, box["y"], 10, key=f"y_p_{i}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with c_w:
                    nw = st.number_input("W", 10, img_w, box["w"], 10, key=f"w_p_{i}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with c_h:
                    nh = st.number_input("H", 10, img_h, box["h"], 10, key=f"h_p_{i}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                if (nx, ny, nw, nh) != (box["x"], box["y"], box["w"], box["h"]):
                    box["x"], box["y"], box["w"], box["h"] = nx, ny, nw, nh
                    changed_manual = True
                with c_del:
                    st.write("")
                    if st.button("Del", key=f"del_p_{i}_{image_id}_{'fs' if is_fullscreen else 'std'}", use_container_width=True):
                        state["pothole_boxes"].pop(i)
                        push_history(image_id, state)
                        st.session_state[canvas_ver_key] += 1
                        save_annotation_from_state(image_id, inv_row, state)
                        st.rerun()

            for j, neg in enumerate(list(state["negative_regions"])):
                stype = neg.get("subtype", "repaired_patch")
                n_lbl, n_x, n_y, n_w, n_h, n_del = st.columns([1.5, 1, 1, 1, 1, 0.8])
                with n_lbl:
                    st.markdown(f"<span style='color:#2563eb; font-weight:600;'>{stype.title()} #{j+1}</span>", unsafe_allow_html=True)
                with n_x:
                    nx = st.number_input("X", 0, img_w, neg["x"], 10, key=f"x_n_{j}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with n_y:
                    ny = st.number_input("Y", 0, img_h, neg["y"], 10, key=f"y_n_{j}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with n_w:
                    nw = st.number_input("W", 10, img_w, neg["w"], 10, key=f"w_n_{j}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                with n_h:
                    nh = st.number_input("H", 10, img_h, neg["h"], 10, key=f"h_n_{j}_{image_id}_{'fs' if is_fullscreen else 'std'}")
                if (nx, ny, nw, nh) != (neg["x"], neg["y"], neg["w"], neg["h"]):
                    neg["x"], neg["y"], neg["w"], neg["h"] = nx, ny, nw, nh
                    changed_manual = True
                with n_del:
                    st.write("")
                    if st.button("Del", key=f"del_n_{j}_{image_id}_{'fs' if is_fullscreen else 'std'}", use_container_width=True):
                        state["negative_regions"].pop(j)
                        push_history(image_id, state)
                        st.session_state[canvas_ver_key] += 1
                        save_annotation_from_state(image_id, inv_row, state)
                        st.rerun()

            if changed_manual:
                push_history(image_id, state)
                st.session_state[canvas_ver_key] += 1
                save_annotation_from_state(image_id, inv_row, state)
                st.rerun()


# ---------------------------------------------------------------------------
# Workspace Dispatch (Fullscreen Mode vs Standard Dual-Panel)
# ---------------------------------------------------------------------------
if is_fullscreen_active:
    # Fullscreen Editable Studio Mode (Spans entire screen width)
    render_canvas_studio(is_fullscreen=True)
else:
    # Standard Dual Panel Workspace (Canvas 70% | Review Station 30%)
    col_viewer, col_actions = st.columns([2.4, 1.0], gap="large")

    with col_viewer:
        render_canvas_studio(is_fullscreen=False)

    with col_actions:
        st.markdown("#### Review Station")

        # One-Click Fast Classification Buttons
        if st.button("Clean Road (0 Potholes)", use_container_width=True):
            state["review_status"] = "reviewed"
            state["patches_present"] = False
            state["pothole_boxes"] = []
            state["uncertain"] = False
            save_annotation_from_state(image_id, inv_row, state)
            if auto_advance and current_idx < len(filtered) - 1:
                st.session_state.selected_image_id = image_ids[current_idx + 1]
            st.rerun()

        if st.button("Repaired Road (0 Potholes)", use_container_width=True):
            state["review_status"] = "reviewed"
            state["patches_present"] = True
            state["pothole_boxes"] = []
            state["uncertain"] = False
            save_annotation_from_state(image_id, inv_row, state)
            if auto_advance and current_idx < len(filtered) - 1:
                st.session_state.selected_image_id = image_ids[current_idx + 1]
            st.rerun()

        if st.button("Mark as Uncertain", use_container_width=True):
            state["review_status"] = "uncertain"
            state["uncertain"] = True
            save_annotation_from_state(image_id, inv_row, state)
            if auto_advance and current_idx < len(filtered) - 1:
                st.session_state.selected_image_id = image_ids[current_idx + 1]
            st.rerun()

        st.markdown("##### Environmental Flags")
        f_patch = st.checkbox("Repaired asphalt patches present", value=state["patches_present"], key=f"f_patch_{image_id}")
        f_shadow = st.checkbox("Shadows across pavement", value=state["shadows_present"], key=f"f_shadow_{image_id}")
        f_manhole = st.checkbox("Manhole / utility cover present", value=state["manholes_present"], key=f"f_manhole_{image_id}")

        if (
            f_patch != state["patches_present"]
            or f_shadow != state["shadows_present"]
            or f_manhole != state["manholes_present"]
        ):
            state["patches_present"] = f_patch
            state["shadows_present"] = f_shadow
            state["manholes_present"] = f_manhole
            save_annotation_from_state(image_id, inv_row, state)
            st.rerun()

        st.markdown("##### Notes")
        notes_val = st.text_input(
            "Notes",
            value=state["notes"],
            placeholder="e.g. wet road, heavy shadows...",
            label_visibility="collapsed",
            key=f"notes_input_{image_id}",
        )
        if notes_val != state["notes"]:
            state["notes"] = notes_val
            save_annotation_from_state(image_id, inv_row, state)

        # Advanced Tools (Collapsed)
        with st.expander("Advanced & Export", expanded=False):
            if st.button("Reset to Unreviewed", use_container_width=True):
                state["review_status"] = "unreviewed"
                save_annotation_from_state(image_id, inv_row, state)
                st.rerun()

            if pil_img is not None:
                buf = io.BytesIO()
                rendered = render_annotated_image(pil_img, state["pothole_boxes"], state["negative_regions"])
                rendered.save(buf, format="JPEG", quality=90)
                st.download_button(
                    "Download Overlaid JPG",
                    data=buf.getvalue(),
                    file_name=f"{image_id}_annotated.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            st.caption(f"Storage: `data/annotations/{'video' if image_id.startswith(('vid_', 'frame_')) else 'green_bay'}/{image_id}.json`")

