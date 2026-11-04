"""Event-level ground-truth evaluation and matching module.

Implements rigorous evaluation matching conforming to methodological standards:
  - Respects video identity (video_id must match).
  - Enforces temporal alignment (observation timestamp within temporal tolerance Δt).
  - Uses physical ground-truth event IDs (gt_event_id) so multiple annotated frames
    of one physical defect do NOT inflate the denominator.
  - Enforces one-to-one report-to-event matching at IoU >= 0.5 (with separate sensitivity at >= 0.3).
  - Prevents non-pothole annotations (shadow, patch, manhole, ordinary road) from counting as TPs.
  - Classifies false reports against negative annotations for look-alike breakdown.
  - Counts duplicate retained reports separately and reports them explicitly.
  - Counts unmatched ground-truth potholes as misses (False Negatives), including defects never proposed.
  - Computes event recall, report precision, false reports per video minute, duplicate counts,
    and look-alike breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthAnnotation:
    """A ground truth bounding-box annotation for evaluation."""
    video_id: str
    timestamp_ms: int
    x1: int
    y1: int
    x2: int
    y2: int
    gt_event_id: str             # Physical defect event ID (same ID across multi-frame views of 1 defect)
    label: str = "pothole"       # "pothole" | "shadow" | "repaired_patch" | "manhole" | "ordinary_road"
    frame_index: Optional[int] = None
    source_image_id: str = ""

    @property
    def is_pothole(self) -> bool:
        return self.label.strip().lower() == "pothole"

    @property
    def box_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class EvaluationMetrics:
    """Summary metrics of event-level evaluation."""
    iou_threshold: float
    total_gt_potholes: int        # Number of unique physical pothole defects in ground truth
    total_reports: int            # Number of candidate reports evaluated
    true_positives: int           # Reports matching a unique physical pothole
    false_positives: int          # Reports not matching any pothole defect
    duplicate_reports: int        # Secondary reports matching an already-claimed physical pothole
    missed_potholes: int          # Physical potholes with no matching report (FN)
    event_recall: float           # TP / total_gt_potholes
    report_precision: float       # TP / (TP + FP + duplicate_reports)
    false_reports_per_minute: float  # (FP + duplicate_reports) / video_duration_min
    lookalike_breakdown: Dict[str, int] = field(default_factory=dict)
    matched_pairs: List[Dict[str, Any]] = field(default_factory=list)
    unmatched_gt_ids: List[str] = field(default_factory=list)
    duplicate_pairs: List[Dict[str, Any]] = field(default_factory=list)


def box_iou(a: Sequence[int | float], b: Sequence[int | float]) -> float:
    """Calculate Intersection-over-Union (IoU) between two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _find_best_gt_match(
    report_obs: Sequence[Any],
    gt_boxes: Sequence[GroundTruthAnnotation],
    video_id: str,
    temporal_tolerance_ms: int,
    iou_threshold: float,
    potholes_only: bool = True,
) -> Tuple[Optional[GroundTruthAnnotation], float]:
    """Find the best ground truth box matching any observation of a candidate report."""
    best_gt = None
    best_iou = 0.0

    for obs in report_obs:
        obs_ts = getattr(obs, "timestamp_ms", 0) if not isinstance(obs, dict) else obs.get("timestamp_ms", 0)
        obs_box = (
            (obs.x1, obs.y1, obs.x2, obs.y2)
            if not isinstance(obs, dict)
            else (obs["x1"], obs["y1"], obs["x2"], obs["y2"])
        )

        for gt in gt_boxes:
            if gt.video_id != video_id:
                continue
            if potholes_only and not gt.is_pothole:
                continue
            if not potholes_only and gt.is_pothole:
                continue

            # Check temporal alignment
            time_diff = abs(obs_ts - gt.timestamp_ms)
            if time_diff > temporal_tolerance_ms:
                continue

            iou = box_iou(obs_box, gt.box_xyxy)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt = gt

    return best_gt, best_iou


def evaluate_event_detections(
    predicted_events: Sequence[Any],
    ground_truth_annotations: Sequence[GroundTruthAnnotation | dict],
    video_duration_min: float = 1.0,
    iou_threshold: float = 0.5,
    temporal_tolerance_ms: int = 1500,
) -> EvaluationMetrics:
    """Perform one-to-one event-level matching and calculate comprehensive evaluation metrics.

    Parameters
    ----------
    predicted_events : sequence of CandidateEvent or dicts
        The retained candidate reports to evaluate.
    ground_truth_annotations : sequence of GroundTruthAnnotation or dicts
        Independent ground-truth boxes with physical gt_event_id.
    video_duration_min : float
        Video duration in minutes for calculating false alert rate.
    iou_threshold : float
        Spatial overlap threshold (default 0.5, sensitivity 0.3).
    temporal_tolerance_ms : int
        Maximum timestamp difference (ms) to consider an observation aligned with GT (default 1500 ms).
    """
    # Normalize ground truth annotations
    gt_list: list[GroundTruthAnnotation] = []
    for g in ground_truth_annotations:
        if isinstance(g, GroundTruthAnnotation):
            gt_list.append(g)
        elif isinstance(g, dict):
            # Parse dict
            gt_list.append(
                GroundTruthAnnotation(
                    video_id=str(g.get("video_id", "")),
                    timestamp_ms=int(g.get("timestamp_ms", 0)),
                    x1=int(g.get("x1", g.get("x", 0))),
                    y1=int(g.get("y1", g.get("y", 0))),
                    x2=int(g.get("x2", g.get("x", 0) + g.get("w", 0))),
                    y2=int(g.get("y2", g.get("y", 0) + g.get("h", 0))),
                    gt_event_id=str(g.get("gt_event_id", g.get("event_id", f"gt_{g.get('timestamp_ms', 0)}"))),
                    label=str(g.get("label", "pothole")),
                    frame_index=g.get("frame_index"),
                    source_image_id=str(g.get("source_image_id", g.get("image_id", ""))),
                )
            )

    # Group unique physical ground-truth potholes by (video_id, gt_event_id)
    unique_gt_potholes: dict[tuple[str, str], list[GroundTruthAnnotation]] = {}
    for gt in gt_list:
        if gt.is_pothole:
            unique_gt_potholes.setdefault((gt.video_id, gt.gt_event_id), []).append(gt)

    total_gt_potholes = len(unique_gt_potholes)

    # Sort reports by confidence descending so strongest detector evidence gets first matching claim
    sorted_reports = list(predicted_events)
    sorted_reports.sort(
        key=lambda ev: getattr(ev, "detector_conf_max", 0.0) if not isinstance(ev, dict) else ev.get("detector_conf_max", 0.0),
        reverse=True,
    )

    claimed_gt_events: set[tuple[str, str]] = set()
    matched_pairs: list[dict] = []
    duplicate_pairs: list[dict] = []
    false_reports: list[dict] = []

    lookalike_counts: dict[str, int] = {
        "shadow": 0,
        "repaired_patch": 0,
        "manhole": 0,
        "ordinary_road": 0,
        "unclassified": 0,
    }

    for ev in sorted_reports:
        ev_id = getattr(ev, "event_id", "") if not isinstance(ev, dict) else ev.get("event_id", "")
        ev_vid = getattr(ev, "video_id", "") if not isinstance(ev, dict) else ev.get("video_id", "")

        # Get observations list or representative box
        obs_list = getattr(ev, "observations", []) if not isinstance(ev, dict) else ev.get("observations", [])
        if not obs_list:
            # Fall back to representative observation
            rep_ts = getattr(ev, "first_timestamp_ms", 0) if not isinstance(ev, dict) else ev.get("first_timestamp_ms", 0)
            rep_x1 = getattr(ev, "rep_x1", 0) if not isinstance(ev, dict) else ev.get("rep_x1", 0)
            rep_y1 = getattr(ev, "rep_y1", 0) if not isinstance(ev, dict) else ev.get("rep_y1", 0)
            rep_x2 = getattr(ev, "rep_x2", 0) if not isinstance(ev, dict) else ev.get("rep_x2", 0)
            rep_y2 = getattr(ev, "rep_y2", 0) if not isinstance(ev, dict) else ev.get("rep_y2", 0)
            obs_list = [{"timestamp_ms": rep_ts, "x1": rep_x1, "y1": rep_y1, "x2": rep_x2, "y2": rep_y2}]

        # 1. Match against true pothole ground truth
        best_gt, best_iou = _find_best_gt_match(
            obs_list, gt_list, ev_vid, temporal_tolerance_ms, iou_threshold, potholes_only=True
        )

        if best_gt is not None:
            gt_key = (best_gt.video_id, best_gt.gt_event_id)
            if gt_key not in claimed_gt_events:
                # First valid report claiming this physical pothole -> True Positive
                claimed_gt_events.add(gt_key)
                matched_pairs.append({
                    "event_id": ev_id,
                    "gt_event_id": best_gt.gt_event_id,
                    "video_id": ev_vid,
                    "iou": best_iou,
                    "matched_gt_timestamp_ms": best_gt.timestamp_ms,
                })
            else:
                # Already claimed by another report -> Duplicate Report
                duplicate_pairs.append({
                    "event_id": ev_id,
                    "gt_event_id": best_gt.gt_event_id,
                    "video_id": ev_vid,
                    "iou": best_iou,
                })
        else:
            # 2. False Report: check if it matched any negative/look-alike annotation
            best_neg, _ = _find_best_gt_match(
                obs_list, gt_list, ev_vid, temporal_tolerance_ms, iou_threshold, potholes_only=False
            )
            cat = best_neg.label.strip().lower() if best_neg else "unclassified"
            if cat not in lookalike_counts:
                cat = "unclassified"
            lookalike_counts[cat] += 1
            false_reports.append({
                "event_id": ev_id,
                "video_id": ev_vid,
                "lookalike_category": cat,
            })

    # Unmatched ground-truth potholes (Misses / False Negatives)
    unmatched_gt_ids = [
        f"{vid}:{gid}" for (vid, gid) in unique_gt_potholes.keys()
        if (vid, gid) not in claimed_gt_events
    ]
    missed_count = len(unmatched_gt_ids)

    tp = len(matched_pairs)
    fp = len(false_reports)
    dup = len(duplicate_pairs)
    total_retained = len(sorted_reports)

    recall = (tp / total_gt_potholes) if total_gt_potholes > 0 else 0.0
    precision = (tp / total_retained) if total_retained > 0 else 0.0

    duration_min = max(0.001, float(video_duration_min))
    false_reports_per_min = (fp + dup) / duration_min

    return EvaluationMetrics(
        iou_threshold=iou_threshold,
        total_gt_potholes=total_gt_potholes,
        total_reports=total_retained,
        true_positives=tp,
        false_positives=fp,
        duplicate_reports=dup,
        missed_potholes=missed_count,
        event_recall=recall,
        report_precision=precision,
        false_reports_per_minute=false_reports_per_min,
        lookalike_breakdown=lookalike_counts,
        matched_pairs=matched_pairs,
        unmatched_gt_ids=unmatched_gt_ids,
        duplicate_pairs=duplicate_pairs,
    )


def run_sensitivity_analysis(
    predicted_events: Sequence[Any],
    ground_truth_annotations: Sequence[GroundTruthAnnotation | dict],
    video_duration_min: float = 1.0,
    thresholds: Sequence[float] = (0.5, 0.3),
    temporal_tolerance_ms: int = 1500,
) -> Dict[float, EvaluationMetrics]:
    """Run event-level evaluation across multiple IoU thresholds (e.g. 0.5 and 0.3)."""
    results: dict[float, EvaluationMetrics] = {}
    for thresh in thresholds:
        results[thresh] = evaluate_event_detections(
            predicted_events=predicted_events,
            ground_truth_annotations=ground_truth_annotations,
            video_duration_min=video_duration_min,
            iou_threshold=thresh,
            temporal_tolerance_ms=temporal_tolerance_ms,
        )
    return results

