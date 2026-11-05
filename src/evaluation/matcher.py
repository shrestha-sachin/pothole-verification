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

import numpy as np

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


def _hungarian_bipartite_matching(weights: np.ndarray) -> list[tuple[int, int]]:
    """Solve maximum weight bipartite matching for N reports and M ground-truth defects.

    Maximizes the total matching weight. Because valid edges are assigned a large
    base weight (1000.0 + IoU), this strictly maximizes the number of one-to-one
    matches, breaking ties by highest total IoU.
    Prefers lap.lapjv (from lapx) for C-speed; falls back to pure Python Kuhn-Munkres.
    """
    N, M = weights.shape
    if N == 0 or M == 0:
        return []

    # Attempt fast C-solver from lapx
    try:
        import lap
        K = max(N, M)
        cost = np.zeros((K, K), dtype=np.float64)
        for i in range(N):
            for j in range(M):
                if weights[i, j] > 0:
                    cost[i, j] = -weights[i, j]
        opt, x, y = lap.lapjv(cost)
        return [(i, int(x[i])) for i in range(N) if x[i] < M and weights[i, x[i]] > 0]
    except Exception:
        pass

    # Pure Python Hungarian (Kuhn-Munkres) O(V^3) algorithm
    K = max(N, M)
    cost = np.zeros((K, K), dtype=np.float64)
    for i in range(N):
        for j in range(M):
            if weights[i, j] > 0:
                cost[i, j] = -weights[i, j]

    u = np.zeros(K + 1)
    v = np.zeros(K + 1)
    p = np.zeros(K + 1, dtype=int)
    way = np.zeros(K + 1, dtype=int)
    for i in range(1, K + 1):
        p[0] = i
        j0 = 0
        minv = np.full(K + 1, np.inf)
        used = np.zeros(K + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, K + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(K + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    ans = [-1] * N
    for j in range(1, K + 1):
        if p[j] <= N:
            ans[p[j] - 1] = j - 1

    return [(i, ans[i]) for i in range(N) if 0 <= ans[i] < M and weights[i, ans[i]] > 0]


def evaluate_event_detections(
    predicted_events: Sequence[Any],
    ground_truth_annotations: Sequence[GroundTruthAnnotation | dict],
    video_duration_min: float = 1.0,
    iou_threshold: float = 0.5,
    temporal_tolerance_ms: int = 1500,
) -> EvaluationMetrics:
    """Perform global one-to-one event-level matching and calculate comprehensive evaluation metrics.

    Uses maximum-weight bipartite matching to globally maximize the number of
    valid one-to-one matches (breaking ties by IoU), resolving greedy assignment
    pathologies.

    Parameters
    ----------
    predicted_events : sequence of CandidateEvent or dicts
        The candidate reports to evaluate.
    ground_truth_annotations : sequence of GroundTruthAnnotation or dicts
        Independent ground-truth boxes with physical gt_event_id.
    video_duration_min : float
        Video duration in minutes for calculating false alert rate (must be positive).
    iou_threshold : float
        Spatial overlap threshold (default 0.5, sensitivity 0.3).
    temporal_tolerance_ms : int
        Maximum timestamp difference (ms) to consider an observation aligned with GT (default 1500 ms).
    """
    if video_duration_min <= 0:
        raise ValueError(f"video_duration_min must be strictly positive (got {video_duration_min} minutes).")

    # Normalize and validate ground truth annotations
    gt_list: list[GroundTruthAnnotation] = []
    for idx, g in enumerate(ground_truth_annotations):
        if isinstance(g, GroundTruthAnnotation):
            if not g.gt_event_id or not str(g.gt_event_id).strip():
                raise ValueError(f"Ground truth annotation at index {idx} missing required physical defect ID 'gt_event_id'.")
            gt_list.append(g)
        elif isinstance(g, dict):
            raw_gid = g.get("gt_event_id") or g.get("event_id")
            if not raw_gid or not str(raw_gid).strip():
                raise ValueError(f"Ground truth dict at index {idx} missing required physical defect ID 'gt_event_id'.")
            gt_list.append(
                GroundTruthAnnotation(
                    video_id=str(g.get("video_id", "")),
                    timestamp_ms=int(g.get("timestamp_ms", 0)),
                    x1=int(g.get("x1", g.get("x", 0))),
                    y1=int(g.get("y1", g.get("y", 0))),
                    x2=int(g.get("x2", g.get("x", 0) + g.get("w", 0))),
                    y2=int(g.get("y2", g.get("y", 0) + g.get("h", 0))),
                    gt_event_id=str(raw_gid).strip(),
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

    gt_pothole_keys = sorted(unique_gt_potholes.keys())
    total_gt_potholes = len(gt_pothole_keys)

    # Extract reports and their observation lists (using representative timestamp if observations empty)
    reports = list(predicted_events)
    report_obs_lists: list[list[dict]] = []

    for ev in reports:
        obs_list = getattr(ev, "observations", None)
        if obs_list is None and isinstance(ev, dict):
            obs_list = ev.get("observations")

        if obs_list:
            report_obs_lists.append([
                {
                    "timestamp_ms": getattr(o, "timestamp_ms", 0) if not isinstance(o, dict) else o.get("timestamp_ms", 0),
                    "frame_index": getattr(o, "frame_index", None) if not isinstance(o, dict) else o.get("frame_index"),
                    "x1": getattr(o, "x1", 0) if not isinstance(o, dict) else o.get("x1", 0),
                    "y1": getattr(o, "y1", 0) if not isinstance(o, dict) else o.get("y1", 0),
                    "x2": getattr(o, "x2", 0) if not isinstance(o, dict) else o.get("x2", 0),
                    "y2": getattr(o, "y2", 0) if not isinstance(o, dict) else o.get("y2", 0),
                }
                for o in obs_list
            ])
        else:
            # Fall back to representative observation, strictly using rep_timestamp_ms if present
            rep_ts = getattr(ev, "rep_timestamp_ms", None)
            if rep_ts is None and isinstance(ev, dict):
                rep_ts = ev.get("rep_timestamp_ms")
            if rep_ts is None:
                rep_ts = getattr(ev, "first_timestamp_ms", 0) if not isinstance(ev, dict) else ev.get("first_timestamp_ms", 0)

            rep_fi = getattr(ev, "rep_frame_index", None)
            if rep_fi is None and isinstance(ev, dict):
                rep_fi = ev.get("rep_frame_index")

            rep_x1 = getattr(ev, "rep_x1", 0) if not isinstance(ev, dict) else ev.get("rep_x1", 0)
            rep_y1 = getattr(ev, "rep_y1", 0) if not isinstance(ev, dict) else ev.get("rep_y1", 0)
            rep_x2 = getattr(ev, "rep_x2", 0) if not isinstance(ev, dict) else ev.get("rep_x2", 0)
            rep_y2 = getattr(ev, "rep_y2", 0) if not isinstance(ev, dict) else ev.get("rep_y2", 0)

            report_obs_lists.append([{
                "timestamp_ms": rep_ts,
                "frame_index": rep_fi,
                "x1": rep_x1,
                "y1": rep_y1,
                "x2": rep_x2,
                "y2": rep_y2,
            }])

    N = len(reports)
    M = total_gt_potholes

    # Construct Bipartite Graph Weight Matrix
    # Large base weight (1000.0) guarantees maximizing cardinality of matching first,
    # and +IoU resolves assignments by highest spatial overlap.
    W = np.zeros((N, M), dtype=np.float64)
    best_matching_obs_info: dict[tuple[int, int], tuple[float, GroundTruthAnnotation]] = {}

    for i, ev in enumerate(reports):
        ev_vid = getattr(ev, "video_id", "") if not isinstance(ev, dict) else ev.get("video_id", "")
        obs_entries = report_obs_lists[i]

        for j, gt_key in enumerate(gt_pothole_keys):
            if ev_vid != gt_key[0]:
                continue
            gt_boxes = unique_gt_potholes[gt_key]

            max_edge_iou = 0.0
            best_gt_box = None

            for obs in obs_entries:
                obs_ts = obs["timestamp_ms"]
                obs_box = (obs["x1"], obs["y1"], obs["x2"], obs["y2"])

                for gt_box in gt_boxes:
                    # Temporal tolerance check
                    if abs(obs_ts - gt_box.timestamp_ms) > temporal_tolerance_ms:
                        continue
                    iou = box_iou(obs_box, gt_box.box_xyxy)
                    if iou >= iou_threshold and iou > max_edge_iou:
                        max_edge_iou = iou
                        best_gt_box = gt_box

            if max_edge_iou >= iou_threshold and best_gt_box is not None:
                W[i, j] = 1000.0 + max_edge_iou
                best_matching_obs_info[(i, j)] = (max_edge_iou, best_gt_box)

    # Solve global optimal assignment
    optimal_matches = _hungarian_bipartite_matching(W)

    matched_report_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    matched_pairs: list[dict] = []

    for r_idx, g_idx in optimal_matches:
        if W[r_idx, g_idx] > 500.0:
            matched_report_indices.add(r_idx)
            matched_gt_indices.add(g_idx)
            edge_iou, matched_gt = best_matching_obs_info[(r_idx, g_idx)]
            ev = reports[r_idx]
            ev_id = getattr(ev, "event_id", "") if not isinstance(ev, dict) else ev.get("event_id", "")
            ev_vid = getattr(ev, "video_id", "") if not isinstance(ev, dict) else ev.get("video_id", "")

            matched_pairs.append({
                "event_id": ev_id,
                "gt_event_id": matched_gt.gt_event_id,
                "video_id": ev_vid,
                "iou": edge_iou,
                "matched_gt_timestamp_ms": matched_gt.timestamp_ms,
            })

    # Unmatched ground-truth potholes are Misses (False Negatives)
    unmatched_gt_ids = [
        f"{gt_pothole_keys[j][0]}:{gt_pothole_keys[j][1]}"
        for j in range(M)
        if j not in matched_gt_indices
    ]
    missed_count = len(unmatched_gt_ids)

    # Classify unmatched reports:
    # If an unmatched report has a valid overlap (IoU >= thresh, time diff <= tol)
    # with an ALREADY MATCHED physical pothole -> Duplicate Report.
    # Otherwise -> False Positive (check negative regions for lookalike breakdown).
    duplicate_pairs: list[dict] = []
    false_reports: list[dict] = []
    lookalike_counts: dict[str, int] = {
        "shadow": 0,
        "repaired_patch": 0,
        "manhole": 0,
        "ordinary_road": 0,
        "unclassified": 0,
    }

    for i, ev in enumerate(reports):
        if i in matched_report_indices:
            continue
        ev_id = getattr(ev, "event_id", "") if not isinstance(ev, dict) else ev.get("event_id", "")
        ev_vid = getattr(ev, "video_id", "") if not isinstance(ev, dict) else ev.get("video_id", "")

        # Check if it overlaps any already-claimed ground-truth pothole
        duplicate_match = None
        for j in matched_gt_indices:
            if W[i, j] > 500.0:
                edge_iou, matched_gt = best_matching_obs_info[(i, j)]
                duplicate_match = (matched_gt, edge_iou)
                break

        if duplicate_match is not None:
            gt_box, d_iou = duplicate_match
            duplicate_pairs.append({
                "event_id": ev_id,
                "gt_event_id": gt_box.gt_event_id,
                "video_id": ev_vid,
                "iou": d_iou,
            })
        else:
            # False Positive: check against negative ground-truth regions
            obs_entries = report_obs_lists[i]
            best_neg, _ = _find_best_gt_match(
                obs_entries, gt_list, ev_vid, temporal_tolerance_ms, iou_threshold, potholes_only=False
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

    tp = len(matched_pairs)
    fp = len(false_reports)
    dup = len(duplicate_pairs)
    total_retained = len(reports)

    recall = (tp / total_gt_potholes) if total_gt_potholes > 0 else 0.0
    precision = (tp / (tp + fp + dup)) if (tp + fp + dup) > 0 else 0.0
    false_reports_per_min = (fp + dup) / video_duration_min

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


def load_video_ground_truth(video_id: str, project_root=None) -> list[GroundTruthAnnotation]:
    """Load explicitly verified annotations with exact video identity and physical IDs.

    Legacy frame annotations remain usable for training, but require explicit
    video_id, timestamp_ms, ground_truth_verified and per-box gt_event_id for scoring.
    No filename inference or automatic physical-defect clustering is performed.
    """
    import json
    from pathlib import Path
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    annotations = []
    for path in sorted((root / "data/annotations/video").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if data.get("video_id") != video_id:
            continue
        if data.get("review_status") != "reviewed" or data.get("uncertain"):
            continue
        if data.get("ground_truth_verified") is not True:
            raise ValueError(f"{path.name}: physical ground-truth IDs must be human verified (ground_truth_verified=true).")
        if "timestamp_ms" not in data:
            raise ValueError(f"{path.name}: explicit timestamp_ms required.")
        rows = []
        for label, boxes in (("pothole", data.get("pothole_boxes", [])),
                             ("negative", data.get("negative_regions", []))):
            for box in boxes:
                rows.append({"video_id": video_id, "timestamp_ms": data["timestamp_ms"],
                             "x": box.get("x"), "y": box.get("y"), "w": box.get("w"), "h": box.get("h"),
                             "gt_event_id": box.get("gt_event_id", ""),
                             "label": "pothole" if label == "pothole" else box.get("subtype", "ordinary_road")})
        import csv, io
        if rows:
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
            try:
                annotations.extend(parse_ground_truth_csv(out.getvalue(), video_id))
            except ValueError as exc:
                raise ValueError(f"{path.name}: {exc}") from exc
    return annotations


def parse_ground_truth_csv(csv_text: str, default_video_id: str = "video_001") -> list[GroundTruthAnnotation]:
    """Strict evaluation import. IDs identify physical defects across annotated frames."""
    import csv, io, math
    annotations = []
    for line, row in enumerate(csv.DictReader(io.StringIO(csv_text)), 2):
        try:
            r = {k.lower().strip(): (v or "").strip() for k, v in row.items() if k}
            vid = r.get("video_id") or default_video_id
            if vid != default_video_id and default_video_id != "video_001":
                raise ValueError("video_id does not match selected video")
            raw_ts = r.get("timestamp_ms")
            ts = float(raw_ts) if raw_ts else float(r["timestamp_sec"]) * 1000
            if not math.isfinite(ts) or ts < 0:
                raise ValueError("timestamp must be finite and nonnegative")
            if all(r.get(k) for k in ("x1", "y1", "x2", "y2")):
                x1,y1,x2,y2 = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            else:
                x1,y1,w,h = [float(r[k]) for k in ("x", "y", "w", "h")]
                x2,y2 = x1+w,y1+h
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or min(x1,y1)<0 or x2<=x1 or y2<=y1:
                raise ValueError("invalid bounding box")
            label = r.get("label") or "pothole"
            gid = r.get("gt_event_id")
            if not gid:
                if label.lower() == "pothole":
                    raise ValueError("gt_event_id required; reuse the same ID for the same physical pothole")
                gid = f"negative_row_{line}"
            annotations.append(GroundTruthAnnotation(video_id=vid, timestamp_ms=int(ts),
                x1=int(x1),y1=int(y1),x2=int(x2),y2=int(y2),gt_event_id=gid,label=label))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"CSV row {line}: {exc}") from exc
    return annotations
