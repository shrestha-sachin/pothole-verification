"""Source-aware road-focused Region of Interest (ROI) preprocessing and optional horizontal tiling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class SourceROI:
    """Normalized [x1, y1, x2, y2] ROI boundaries in [0, 1] for a camera source."""
    x1: float
    y1: float
    x2: float
    y2: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        px1 = max(0, min(width - 1, int(round(self.x1 * width))))
        py1 = max(0, min(height - 1, int(round(self.y1 * height))))
        px2 = max(px1 + 1, min(width, int(round(self.x2 * width))))
        py2 = max(py1 + 1, min(height, int(round(self.y2 * height))))
        return px1, py1, px2, py2


# Calibrated from annotated training box boundaries + horizon safety margins:
# - local_dashcam: removes top 64% sky/trees/powerlines/buildings, trims roadside edges & bottom timestamp border
# - local_smartphone: removes top 46% sky/storefronts while keeping full horizon road view
# - mapillary: removes top 50% sky/upper buildings across 4:3 street-view shots
# - rdd2022_usa: removes top 45% sky/horizon in 640x640 shots
DEFAULT_SOURCE_ROIS: dict[str, SourceROI] = {
    "local_dashcam": SourceROI(x1=0.04, y1=0.64, x2=0.88, y2=0.992),
    "local_smartphone": SourceROI(x1=0.04, y1=0.46, x2=0.96, y2=0.992),
    "mapillary": SourceROI(x1=0.00, y1=0.50, x2=1.00, y2=1.000),
    "rdd2022_usa": SourceROI(x1=0.00, y1=0.45, x2=1.00, y2=1.000),
}


@dataclass
class CropTransformResult:
    image: Image.Image
    crop_box_px: tuple[int, int, int, int]
    orig_wh: tuple[int, int]
    crop_wh: tuple[int, int]
    boxes: list[tuple[int, float, float, float, float]]
    original_box_count: int
    dropped_boxes: int
    truncated_boxes: int


def transform_boxes_to_crop(
    boxes_norm: list[tuple[int, float, float, float, float]],
    orig_w: int,
    orig_h: int,
    crop_px: tuple[int, int, int, int],
    *,
    min_visibility: float = 0.35,
) -> tuple[list[tuple[int, float, float, float, float]], int, int]:
    """Translate normalized YOLO boxes into a crop rectangle. Returns (boxes, dropped, truncated)."""
    cx1, cy1, cx2, cy2 = crop_px
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    out: list[tuple[int, float, float, float, float]] = []
    dropped = 0
    truncated = 0

    for cls_id, cx, cy, bw, bh in boxes_norm:
        bx1 = (cx - bw / 2.0) * orig_w
        by1 = (cy - bh / 2.0) * orig_h
        bx2 = (cx + bw / 2.0) * orig_w
        by2 = (cy + bh / 2.0) * orig_h
        orig_area = max(1e-6, (bx2 - bx1) * (by2 - by1))

        ix1 = max(float(cx1), bx1)
        iy1 = max(float(cy1), by1)
        ix2 = min(float(cx2), bx2)
        iy2 = min(float(cy2), by2)
        iw_clipped = max(0.0, ix2 - ix1)
        ih_clipped = max(0.0, iy2 - iy1)
        vis_ratio = (iw_clipped * ih_clipped) / orig_area

        if iw_clipped < 2.0 or ih_clipped < 2.0 or vis_ratio < min_visibility:
            dropped += 1
            continue
        if vis_ratio < 0.98:
            truncated += 1

        ncx = ((ix1 + ix2) / 2.0 - cx1) / cw
        ncy = ((iy1 + iy2) / 2.0 - cy1) / ch
        nbw = iw_clipped / cw
        nbh = ih_clipped / ch
        out.append((cls_id, round(ncx, 6), round(ncy, 6), round(nbw, 6), round(nbh, 6)))

    return out, dropped, truncated


def apply_roi_crop(
    img: Image.Image,
    boxes_norm: list[tuple[int, float, float, float, float]],
    roi: SourceROI,
    *,
    min_visibility: float = 0.35,
) -> CropTransformResult:
    """Apply deterministic ROI crop to image and translate bounding boxes."""
    orig_w, orig_h = img.size
    crop_px = roi.to_pixels(orig_w, orig_h)
    cropped = img.crop(crop_px)
    new_boxes, dropped, truncated = transform_boxes_to_crop(
        boxes_norm, orig_w, orig_h, crop_px, min_visibility=min_visibility
    )
    return CropTransformResult(
        image=cropped,
        crop_box_px=crop_px,
        orig_wh=(orig_w, orig_h),
        crop_wh=cropped.size,
        boxes=new_boxes,
        original_box_count=len(boxes_norm),
        dropped_boxes=dropped,
        truncated_boxes=truncated,
    )


def _horizontal_tile_offsets(
    cw: int,
    ch: int,
    *,
    overlap_frac: float = 0.15,
    min_tile_trigger_width: int = 2000,
) -> list[tuple[int, int, int, int]]:
    """Compute local horizontal tile rectangles inside an ROI of size (cw, ch)."""
    if cw < min_tile_trigger_width:
        return [(0, 0, cw, ch)]
    half = int(round(cw * (0.5 + overlap_frac / 2.0)))
    return [
        (0, 0, half, ch),
        (max(0, cw - half), 0, cw, ch),
    ]


def compute_roi_and_tile_rects(
    width: int,
    height: int,
    roi: SourceROI | None = None,
    *,
    tile_wide_roi: bool = False,
    overlap_frac: float = 0.15,
    min_tile_trigger_width: int = 2000,
) -> list[tuple[int, int, int, int]]:
    """Return crop/tile rectangles in original-frame pixel coordinates matching dataset curation."""
    if roi is None:
        bx1, by1, bx2, by2 = 0, 0, width, height
    else:
        bx1, by1, bx2, by2 = roi.to_pixels(width, height)

    cw, ch = max(1, bx2 - bx1), max(1, by2 - by1)
    if not tile_wide_roi:
        return [(bx1, by1, bx2, by2)]

    return [
        (bx1 + tx1, by1 + ty1, bx1 + tx2, by1 + ty2)
        for tx1, ty1, tx2, ty2 in _horizontal_tile_offsets(
            cw, ch, overlap_frac=overlap_frac, min_tile_trigger_width=min_tile_trigger_width
        )
    ]


def generate_horizontal_tiles(
    img: Image.Image,
    boxes_norm: list[tuple[int, float, float, float, float]],
    base_crop_px: tuple[int, int, int, int],
    *,
    overlap_frac: float = 0.15,
    min_tile_trigger_width: int = 2000,
    min_visibility: float = 0.35,
) -> list[CropTransformResult]:
    """Optionally split wide high-res road ROIs into 2 overlapping horizontal tiles for small-object boost."""
    cw, ch = img.size
    tile_rects = _horizontal_tile_offsets(
        cw, ch, overlap_frac=overlap_frac, min_tile_trigger_width=min_tile_trigger_width
    )
    if len(tile_rects) == 1:
        return [
            CropTransformResult(
                image=img,
                crop_box_px=base_crop_px,
                orig_wh=img.size,
                crop_wh=img.size,
                boxes=boxes_norm,
                original_box_count=len(boxes_norm),
                dropped_boxes=0,
                truncated_boxes=0,
            )
        ]

    bx1_base, by1_base, _, _ = base_crop_px
    tiles: list[CropTransformResult] = []
    for tx1, ty1, tx2, ty2 in tile_rects:
        t_img = img.crop((tx1, ty1, tx2, ty2))
        t_boxes, dropped, truncated = transform_boxes_to_crop(
            boxes_norm, cw, ch, (tx1, ty1, tx2, ty2), min_visibility=min_visibility
        )
        # Global pixel offset back to original pre-ROI image!
        abs_rect = (bx1_base + tx1, by1_base + ty1, bx1_base + tx2, by1_base + ty2)
        tiles.append(
            CropTransformResult(
                image=t_img,
                crop_box_px=abs_rect,
                orig_wh=(cw, ch),
                crop_wh=t_img.size,
                boxes=t_boxes,
                original_box_count=len(boxes_norm),
                dropped_boxes=dropped,
                truncated_boxes=truncated,
            )
        )
    return tiles


def map_predictions_to_original(
    predictions: list[dict[str, Any]],
    crop_box_px: tuple[int, int, int, int],
    *,
    nms_iou_thresh: float = 0.50,
) -> list[dict[str, Any]]:
    """Map predicted [x1, y1, x2, y2] boxes from ROI/tile coordinates back to original frame coordinates
    and optionally deduplicate overlapping tile predictions with NMS."""
    ox1, oy1, _, _ = crop_box_px
    mapped: list[dict[str, Any]] = []
    for pred in predictions:
        copy = dict(pred)
        copy["x1"] = float(pred["x1"]) + ox1
        copy["y1"] = float(pred["y1"]) + oy1
        copy["x2"] = float(pred["x2"]) + ox1
        copy["y2"] = float(pred["y2"]) + oy1
        mapped.append(copy)

    if len(mapped) <= 1:
        return mapped

    # Greedy NMS across overlapping predictions
    mapped.sort(key=lambda p: float(p.get("conf", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    for cand in mapped:
        keep = True
        for k in kept:
            ix1 = max(cand["x1"], k["x1"])
            iy1 = max(cand["y1"], k["y1"])
            ix2 = min(cand["x2"], k["x2"])
            iy2 = min(cand["y2"], k["y2"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a1 = max(1e-6, (cand["x2"] - cand["x1"]) * (cand["y2"] - cand["y1"]))
            a2 = max(1e-6, (k["x2"] - k["x1"]) * (k["y2"] - k["y1"]))
            if inter / (a1 + a2 - inter) >= nms_iou_thresh:
                keep = False
                break
        if keep:
            kept.append(cand)
    return kept
