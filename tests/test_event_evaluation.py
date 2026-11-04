"""Test event-level evaluation matcher, physical defect IDs, duplicate accounting, and look-alike categorization."""

import unittest
from src.evaluation.matcher import (
    GroundTruthAnnotation,
    evaluate_event_detections,
    run_sensitivity_analysis,
)
from src.tracking.event_grouper import CandidateEvent, ObservedDetection


def make_eval_event(
    event_id: str,
    video_id: str,
    timestamp_ms: int,
    box: tuple[int, int, int, int],
    conf: float = 0.8,
) -> CandidateEvent:
    x1, y1, x2, y2 = box
    obs = ObservedDetection(
        frame_index=int(timestamp_ms / 200),
        timestamp_ms=timestamp_ms,
        x1=x1, y1=y1, x2=x2, y2=y2,
        conf=conf,
        laplacian_var=120.0,
    )
    return CandidateEvent(
        event_id=event_id,
        video_id=video_id,
        road_segment_id="seg_001",
        track_id=1,
        first_timestamp_ms=timestamp_ms,
        last_timestamp_ms=timestamp_ms,
        first_frame_index=obs.frame_index,
        last_frame_index=obs.frame_index,
        detector_conf_max=conf,
        detector_conf_mean=conf,
        observation_count=1,
        rep_x1=x1, rep_y1=y1, rep_x2=x2, rep_y2=y2,
        rep_timestamp_ms=timestamp_ms,
        rep_frame_index=obs.frame_index,
        observations=[obs],
    )


class TestEventLevelEvaluation(unittest.TestCase):
    """Rigorous tests for event-level evaluation."""

    def test_wrong_video_id_is_rejected(self):
        """A report in video_A must never match a GT defect in video_B."""
        report = make_eval_event("rep1", video_id="video_A", timestamp_ms=5000, box=(100, 100, 200, 200))
        gt = GroundTruthAnnotation(
            video_id="video_B", timestamp_ms=5000, x1=100, y1=100, x2=200, y2=200,
            gt_event_id="defect_1", label="pothole",
        )

        res = evaluate_event_detections([report], [gt], video_duration_min=1.0)
        self.assertEqual(res.true_positives, 0)
        self.assertEqual(res.false_positives, 1)
        self.assertEqual(res.missed_potholes, 1)
        self.assertEqual(res.event_recall, 0.0)

    def test_wrong_timestamp_is_rejected(self):
        """A report at t=10.0s must not match a GT defect at t=1.0s even with identical box."""
        report = make_eval_event("rep1", video_id="vid1", timestamp_ms=10000, box=(100, 100, 200, 200))
        gt = GroundTruthAnnotation(
            video_id="vid1", timestamp_ms=1000, x1=100, y1=100, x2=200, y2=200,
            gt_event_id="defect_1", label="pothole",
        )

        res = evaluate_event_detections([report], [gt], video_duration_min=1.0, temporal_tolerance_ms=1500)
        self.assertEqual(res.true_positives, 0)
        self.assertEqual(res.false_positives, 1)
        self.assertEqual(res.missed_potholes, 1)

    def test_repeated_annotations_of_one_physical_pothole_do_not_inflate_denominator(self):
        """CRITICAL: If one physical pothole is annotated across 5 video frames, total GT count must be 1."""
        gt_boxes = [
            GroundTruthAnnotation(
                video_id="vid1", timestamp_ms=t, x1=100, y1=100, x2=200, y2=200,
                gt_event_id="physical_pothole_01", label="pothole",
            )
            for t in [1000, 1200, 1400, 1600, 1800]
        ]

        report = make_eval_event("rep1", video_id="vid1", timestamp_ms=1400, box=(105, 105, 195, 195))

        res = evaluate_event_detections([report], gt_boxes, video_duration_min=1.0)
        self.assertEqual(res.total_gt_potholes, 1)  # Denominator is 1, not 5!
        self.assertEqual(res.true_positives, 1)
        self.assertEqual(res.missed_potholes, 0)
        self.assertEqual(res.false_positives, 0)
        self.assertEqual(res.event_recall, 1.0)     # 1 / 1 = 100%

    def test_non_pothole_annotations_cannot_count_as_true_positives(self):
        """A detection matching a shadow or repaired patch must be recorded as False Positive."""
        report = make_eval_event("rep_shadow", video_id="vid1", timestamp_ms=3000, box=(100, 100, 200, 200))
        gt_shadow = GroundTruthAnnotation(
            video_id="vid1", timestamp_ms=3000, x1=100, y1=100, x2=200, y2=200,
            gt_event_id="shadow_01", label="shadow",
        )

        res = evaluate_event_detections([report], [gt_shadow], video_duration_min=1.0)
        self.assertEqual(res.total_gt_potholes, 0)  # No real potholes in ground truth
        self.assertEqual(res.true_positives, 0)     # Shadow never counts as TP!
        self.assertEqual(res.false_positives, 1)
        self.assertEqual(res.lookalike_breakdown.get("shadow"), 1)

    def test_duplicate_reports_counted_separately(self):
        """When two separate reports match the same physical defect, one is TP and one is Duplicate."""
        gt = GroundTruthAnnotation(
            video_id="vid1", timestamp_ms=5000, x1=100, y1=100, x2=200, y2=200,
            gt_event_id="physical_pothole_01", label="pothole",
        )

        rep1 = make_eval_event("rep1", video_id="vid1", timestamp_ms=5000, box=(100, 100, 200, 200), conf=0.9)
        rep2 = make_eval_event("rep2", video_id="vid1", timestamp_ms=5100, box=(105, 105, 205, 205), conf=0.7)

        res = evaluate_event_detections([rep1, rep2], [gt], video_duration_min=2.0)
        self.assertEqual(res.total_gt_potholes, 1)
        self.assertEqual(res.true_positives, 1)
        self.assertEqual(res.duplicate_reports, 1)
        self.assertEqual(res.false_positives, 0)
        self.assertEqual(res.missed_potholes, 0)
        self.assertEqual(res.event_recall, 1.0)
        self.assertEqual(len(res.duplicate_pairs), 1)
        self.assertEqual(res.duplicate_pairs[0]["event_id"], "rep2")

    def test_completely_missed_pothole_counted_as_fn(self):
        """Potholes present in ground truth that received no candidate reports must be misses."""
        gt1 = GroundTruthAnnotation("vid1", 1000, 50, 50, 150, 150, "p1", "pothole")
        gt2 = GroundTruthAnnotation("vid1", 8000, 300, 300, 400, 400, "p2", "pothole")

        rep1 = make_eval_event("rep1", "vid1", 1000, (50, 50, 150, 150))

        res = evaluate_event_detections([rep1], [gt1, gt2], video_duration_min=1.0)
        self.assertEqual(res.total_gt_potholes, 2)
        self.assertEqual(res.true_positives, 1)
        self.assertEqual(res.missed_potholes, 1)
        self.assertEqual(res.unmatched_gt_ids, ["vid1:p2"])
        self.assertEqual(res.event_recall, 0.5)

    def test_iou_sensitivity_analysis_at_05_and_03(self):
        """Sensitivity analysis: box with IoU = 0.40 fails at 0.5 but matches at 0.3."""
        gt = GroundTruthAnnotation("vid1", 2000, 0, 0, 100, 100, "p1", "pothole")
        rep = make_eval_event("rep1", "vid1", 2000, (0, 0, 100, 250))

        results = run_sensitivity_analysis([rep], [gt], thresholds=(0.5, 0.3))
        self.assertEqual(results[0.5].true_positives, 0)  # IoU 0.40 < 0.5
        self.assertEqual(results[0.3].true_positives, 1)  # IoU 0.40 >= 0.3


if __name__ == "__main__":
    unittest.main()
