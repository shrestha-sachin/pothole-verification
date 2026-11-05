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

    def test_global_bipartite_matching_resolves_greedy_assignment_pathology(self):
        """CRITICAL REGRESSION TEST:
        Report 1 matches Pothole A (IoU 0.8) and Pothole B (IoU 0.6).
        Report 2 only matches Pothole A (IoU 0.7).
        Greedy local matching causes:
          rep1 -> A (claimed)
          rep2 -> A (duplicate!), B is missed (FN!) -> 1 TP, 1 dup, 1 miss.
        Global matching MUST resolve:
          rep1 -> B, rep2 -> A -> 2 True Positives, 0 duplicates, 0 misses!
        """
        # Pothole A at (0, 0, 100, 100)
        gt_a = GroundTruthAnnotation("vid1", 1000, 0, 0, 100, 100, "pothole_A", "pothole")
        # Pothole B at (300, 300, 400, 400)
        gt_b = GroundTruthAnnotation("vid1", 1000, 300, 300, 400, 400, "pothole_B", "pothole")

        # Report 1 has large box overlapping both A (high overlap) and B (moderate overlap)
        # We can construct observations for rep1 that match both A and B
        obs_rep1_a = ObservedDetection(0, 1000, 0, 0, 100, 125, 0.90, 150.0)    # IoU ~0.80 with A
        obs_rep1_b = ObservedDetection(0, 1000, 300, 300, 400, 460, 0.90, 150.0) # IoU ~0.62 with B

        rep1 = CandidateEvent(
            event_id="rep1", video_id="vid1", road_segment_id="seg1", track_id=1,
            first_timestamp_ms=1000, last_timestamp_ms=1000,
            first_frame_index=0, last_frame_index=0,
            detector_conf_max=0.90, detector_conf_mean=0.90, observation_count=2,
            rep_x1=0, rep_y1=0, rep_x2=100, rep_y2=125,
            rep_timestamp_ms=1000, rep_frame_index=0,
            observations=[obs_rep1_a, obs_rep1_b],
        )

        # Report 2 only has observation matching A
        obs_rep2_a = ObservedDetection(0, 1000, 0, 0, 100, 142, 0.75, 120.0)    # IoU ~0.70 with A
        rep2 = CandidateEvent(
            event_id="rep2", video_id="vid1", road_segment_id="seg1", track_id=2,
            first_timestamp_ms=1000, last_timestamp_ms=1000,
            first_frame_index=0, last_frame_index=0,
            detector_conf_max=0.75, detector_conf_mean=0.75, observation_count=1,
            rep_x1=0, rep_y1=0, rep_x2=100, rep_y2=142,
            rep_timestamp_ms=1000, rep_frame_index=0,
            observations=[obs_rep2_a],
        )

        # Forward order [rep1, rep2]
        res = evaluate_event_detections([rep1, rep2], [gt_a, gt_b], video_duration_min=1.0)
        self.assertEqual(res.true_positives, 2, "Must produce 2 True Positives via global matching!")
        self.assertEqual(res.duplicate_reports, 0, "Must produce 0 Duplicates!")
        self.assertEqual(res.missed_potholes, 0, "Must produce 0 Misses!")
        self.assertEqual(res.event_recall, 1.0)
        self.assertEqual(res.report_precision, 1.0)

        # Reversed order [rep2, rep1] — must be 100% order invariant!
        res_rev = evaluate_event_detections([rep2, rep1], [gt_a, gt_b], video_duration_min=1.0)
        self.assertEqual(res_rev.true_positives, 2, "Reversed order must produce 2 True Positives!")
        self.assertEqual(res_rev.duplicate_reports, 0, "Reversed order must produce 0 Duplicates!")
        self.assertEqual(res_rev.missed_potholes, 0, "Reversed order must produce 0 Misses!")
        self.assertEqual(res_rev.event_recall, 1.0)
        self.assertEqual(res_rev.report_precision, 1.0)

    def test_non_positive_duration_raises_error(self):
        gt = GroundTruthAnnotation("vid1", 1000, 0, 0, 50, 50, "p1", "pothole")
        rep = make_eval_event("rep1", "vid1", 1000, (0, 0, 50, 50))
        with self.assertRaises(ValueError):
            evaluate_event_detections([rep], [gt], video_duration_min=0.0)
        with self.assertRaises(ValueError):
            evaluate_event_detections([rep], [gt], video_duration_min=-1.5)

    def test_missing_gt_event_id_raises_error(self):
        gt_invalid = GroundTruthAnnotation("vid1", 1000, 0, 0, 50, 50, "", "pothole")
        rep = make_eval_event("rep1", "vid1", 1000, (0, 0, 50, 50))
        with self.assertRaises(ValueError):
            evaluate_event_detections([rep], [gt_invalid], video_duration_min=1.0)


if __name__ == "__main__":
    unittest.main()
