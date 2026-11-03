"""Reporting policy application module.

Three policies operate on the same cached detections/events:

  Policy A — Detector only
    Retain events where detector_conf_max >= conf_threshold.

  Policy B — Temporal verification
    Retain events where observation_count >= min_obs
    (AND optionally frame_span <= max_frame_span for burst requirement).

  Policy C — Visual verification
    Retain events where policy_c_score >= verifier_threshold.
    Policy C does NOT automatically require Policy B's temporal condition.

Rules:
  - All three policies share identical sampling, tracking, and event grouping.
  - Never retrain YOLO per policy.
  - Thresholds are configuration inputs, not selected from test data.
  - Performance metrics are only computed when independent ground truth is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyConfig:
    """Threshold configuration for all three policies."""
    # Policy A
    conf_threshold: float = 0.35

    # Policy B
    min_obs: int = 2
    max_frame_span: int = 5          # max frames between first and last obs

    # Policy C
    verifier_threshold: float = 0.5  # NOT selected from test data

    # Policy C requires Policy B?  (per spec: No)
    policy_c_requires_policy_b: bool = False


def apply_policies(
    events: list,
    config: Optional[PolicyConfig] = None,
) -> list:
    """
    Annotate each CandidateEvent with policy flags in-place.

    Parameters
    ----------
    events : list of CandidateEvent (from src.tracking.event_grouper)
    config : PolicyConfig (uses defaults if None)

    Returns the same list for chaining.
    """
    cfg = config or PolicyConfig()

    for ev in events:
        # Policy A
        ev.policy_a_retained = ev.detector_conf_max >= cfg.conf_threshold

        # Policy B
        frame_span = ev.last_frame_index - ev.first_frame_index
        ev.policy_b_retained = (
            ev.observation_count >= cfg.min_obs
        )

        # Policy C — only update retained flag if score exists
        if ev.policy_c_score is not None:
            ev.policy_c_retained = ev.policy_c_score >= cfg.verifier_threshold
            if cfg.policy_c_requires_policy_b:
                ev.policy_c_retained = ev.policy_c_retained and ev.policy_b_retained
        else:
            ev.policy_c_retained = None

    return events


def summarize_policies(events: list, config: Optional[PolicyConfig] = None) -> dict:
    """Return per-policy counts for dashboard display."""
    cfg = config or PolicyConfig()
    total = len(events)
    a = sum(1 for e in events if e.policy_a_retained)
    b = sum(1 for e in events if e.policy_b_retained)
    c_scored = sum(1 for e in events if e.policy_c_score is not None)
    c_retained = sum(
        1 for e in events
        if e.policy_c_retained is True
    )

    return {
        "total_candidates": total,
        "policy_a_retained": a,
        "policy_a_filtered": total - a,
        "policy_b_retained": b,
        "policy_b_filtered": total - b,
        "policy_c_scored": c_scored,
        "policy_c_retained": c_retained,
        "policy_c_filtered": c_scored - c_retained,
        "verifier_not_run": total - c_scored,
    }


# ---------------------------------------------------------------------------
# Ground-truth matching (stub — requires external annotations)
# ---------------------------------------------------------------------------

def match_events_to_ground_truth(
    predicted_events: list,
    ground_truth_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """
    One-to-one event matching using IoU >= iou_threshold.

    Parameters
    ----------
    predicted_events : list of CandidateEvent
    ground_truth_boxes : list of dicts with keys:
        {timestamp_ms, x1, y1, x2, y2, label}  # label: pothole | non_pothole
    iou_threshold : default 0.5; use 0.3 for sensitivity analysis

    Returns
    -------
    dict with keys: tp, fp, fn, matched_pairs, unmatched_pred, unmatched_gt

    NOTE: This function is a stub.  It computes correct IoU matching logic
    but should only be called when ground-truth annotations are available.
    Calling it on development/validation data for threshold selection
    would constitute test-set leakage.
    """
    import warnings
    warnings.warn(
        "match_events_to_ground_truth requires independent ground-truth annotations. "
        "Do not use validation or test footage for threshold selection.",
        UserWarning,
        stacklevel=2,
    )

    matched_pred: set[str] = set()
    matched_gt: set[int] = set()
    matched_pairs: list = []

    for ev in predicted_events:
        best_iou = 0.0
        best_gt_idx = -1
        for gi, gt in enumerate(ground_truth_boxes):
            if gi in matched_gt:
                continue
            iou = _box_iou(
                (ev.rep_x1, ev.rep_y1, ev.rep_x2, ev.rep_y2),
                (gt["x1"], gt["y1"], gt["x2"], gt["y2"]),
            )
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_pred.add(ev.event_id)
            matched_gt.add(best_gt_idx)
            matched_pairs.append({
                "event_id": ev.event_id,
                "gt_index": best_gt_idx,
                "iou": best_iou,
            })

    tp = len(matched_pairs)
    fp = len(predicted_events) - tp
    fn = len(ground_truth_boxes) - tp

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matched_pairs": matched_pairs,
        "unmatched_pred": [
            ev.event_id for ev in predicted_events
            if ev.event_id not in matched_pred
        ],
        "unmatched_gt": [
            gi for gi in range(len(ground_truth_boxes))
            if gi not in matched_gt
        ],
    }


def _box_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
