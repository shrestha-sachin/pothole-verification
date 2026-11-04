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

    # Policy B: at least min_obs observations within a sliding window of temporal_window_frames
    min_obs: int = 2
    temporal_window_frames: int = 5   # sliding window size (5 sampled frames)

    # Policy C
    verifier_threshold: float = 0.5   # NOT selected from test data

    # Policy C requires Policy B?  (per spec: No, C is independent of B by default)
    policy_c_requires_policy_b: bool = False


def check_temporal_window(
    observations: list,
    min_obs: int = 2,
    window_size_frames: int = 5,
    conf_threshold: float = 0.0,
) -> bool:
    """Check if at least min_obs actual detector observations occur within any sliding window of window_size_frames.

    Frame indices are sequential sampled frame counters: 0, 1, 2, ...
    A sliding window of size W frames spans indices [k, k + W - 1], so the maximum difference
    between two observation indices in the same window is (W - 1).
    For W = 5, max allowed difference between any two observations in the window is 4.
    """
    if not observations or min_obs < 1:
        return False

    indices = []
    for obs in observations:
        c = getattr(obs, "conf", None) if not isinstance(obs, dict) else obs.get("conf")
        if c is not None and c < conf_threshold:
            continue
        fi = getattr(obs, "frame_index", None) if not isinstance(obs, dict) else obs.get("frame_index")
        if fi is not None:
            indices.append(int(fi))

    if len(indices) < min_obs:
        return False

    indices = sorted(set(indices))
    if len(indices) < min_obs:
        return False

    max_gap = window_size_frames - 1
    for i in range(len(indices) - min_obs + 1):
        if (indices[i + min_obs - 1] - indices[i]) <= max_gap:
            return True
    return False


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
        # Policy A: detector confidence threshold + shared grouping
        max_conf = getattr(ev, "detector_conf_max", 0.0) or 0.0
        ev.policy_a_retained = bool(max_conf >= cfg.conf_threshold)

        # Policy B: Policy A AND at least min_obs actual observations within sliding window
        obs_list = getattr(ev, "observations", []) or []
        temporal_pass = check_temporal_window(
            obs_list,
            min_obs=cfg.min_obs,
            window_size_frames=cfg.temporal_window_frames,
            conf_threshold=cfg.conf_threshold,
        )
        ev.policy_b_retained = bool(ev.policy_a_retained and temporal_pass)

        # Policy C: Policy A AND MobileNetV3-Small score above verifier_threshold.
        # Per specification: Policy C must NOT require Policy B unless explicitly configured.
        if ev.policy_c_score is not None:
            c_pass = ev.policy_c_score >= cfg.verifier_threshold
            if cfg.policy_c_requires_policy_b:
                ev.policy_c_retained = bool(ev.policy_a_retained and ev.policy_b_retained and c_pass)
            else:
                ev.policy_c_retained = bool(ev.policy_a_retained and c_pass)
        else:
            # Policy C unavailable when no valid checkpoint exists; never fabricate scores
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
# Ground-truth matching
# ---------------------------------------------------------------------------

from src.evaluation.matcher import (
    GroundTruthAnnotation,
    EvaluationMetrics,
    evaluate_event_detections,
    run_sensitivity_analysis,
    box_iou,
)


def match_events_to_ground_truth(
    predicted_events: list,
    ground_truth_boxes: list[dict],
    video_duration_min: float = 1.0,
    iou_threshold: float = 0.5,
    temporal_tolerance_ms: int = 1500,
) -> dict:
    """
    One-to-one event matching using physical event IDs, temporal alignment, and IoU.

    Parameters
    ----------
    predicted_events : list of CandidateEvent
    ground_truth_boxes : list of dicts or GroundTruthAnnotation
    video_duration_min : video duration in minutes for false alarm rate calculation
    iou_threshold : default 0.5; use 0.3 for sensitivity analysis
    temporal_tolerance_ms : timestamp tolerance in ms for observation alignment

    Returns
    -------
    dict with keys: tp, fp, fn, duplicate_reports, recall, precision,
                    false_reports_per_minute, lookalike_breakdown, matched_pairs
    """
    metrics = evaluate_event_detections(
        predicted_events=predicted_events,
        ground_truth_annotations=ground_truth_boxes,
        video_duration_min=video_duration_min,
        iou_threshold=iou_threshold,
        temporal_tolerance_ms=temporal_tolerance_ms,
    )
    return {
        "tp": metrics.true_positives,
        "fp": metrics.false_positives,
        "fn": metrics.missed_potholes,
        "duplicate_reports": metrics.duplicate_reports,
        "recall": metrics.event_recall,
        "precision": metrics.report_precision,
        "false_reports_per_minute": metrics.false_reports_per_minute,
        "lookalike_breakdown": metrics.lookalike_breakdown,
        "matched_pairs": metrics.matched_pairs,
        "duplicate_pairs": metrics.duplicate_pairs,
        "unmatched_gt_ids": metrics.unmatched_gt_ids,
        "metrics_object": metrics,
    }
