"""Test reporting policies A, B, C, temporal sliding window boundaries, and policy independence."""

import unittest
from src.evaluation.policies import PolicyConfig, apply_policies, check_temporal_window
from src.tracking.event_grouper import CandidateEvent, ObservedDetection


def make_test_event(
    event_id: str,
    conf_max: float,
    observations: list[tuple[int, int, float]],  # (frame_index, timestamp_ms, conf)
    policy_c_score: float | None = None,
) -> CandidateEvent:
    """Helper to create a synthetic CandidateEvent fixture."""
    obs_objs = [
        ObservedDetection(
            frame_index=fi,
            timestamp_ms=ts,
            x1=100, y1=100, x2=200, y2=200,
            conf=conf,
            laplacian_var=100.0,
        )
        for fi, ts, conf in observations
    ]
    confs = [o.conf for o in obs_objs] if obs_objs else [conf_max]
    first_ts = obs_objs[0].timestamp_ms if obs_objs else 0
    last_ts = obs_objs[-1].timestamp_ms if obs_objs else 0
    first_fi = obs_objs[0].frame_index if obs_objs else 0
    last_fi = obs_objs[-1].frame_index if obs_objs else 0

    return CandidateEvent(
        event_id=event_id,
        video_id="test_vid",
        road_segment_id="seg_001",
        track_id=1,
        first_timestamp_ms=first_ts,
        last_timestamp_ms=last_ts,
        first_frame_index=first_fi,
        last_frame_index=last_fi,
        detector_conf_max=conf_max,
        detector_conf_mean=sum(confs) / len(confs),
        observation_count=len(obs_objs),
        rep_x1=100, rep_y1=100, rep_x2=200, rep_y2=200,
        rep_timestamp_ms=first_ts,
        rep_frame_index=first_fi,
        observations=obs_objs,
        policy_c_score=policy_c_score,
    )


class TestTemporalSlidingWindow(unittest.TestCase):
    """Tests for the 5-sampled-frame sliding window temporal rule (Policy B)."""

    def test_single_observation_fails_policy_b(self):
        obs = [ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.8)]
        self.assertFalse(check_temporal_window(obs, min_obs=2, window_size_frames=5))

    def test_boundary_within_five_frames_passes(self):
        # Frame indices 0 and 4: difference is 4 frames -> inside window [0..4] (5 frames total)
        obs = [
            ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=4, timestamp_ms=800, x1=0, y1=0, x2=10, y2=10, conf=0.85),
        ]
        self.assertTrue(check_temporal_window(obs, min_obs=2, window_size_frames=5))

    def test_boundary_exceeding_five_frames_fails(self):
        # Frame indices 0 and 5: difference is 5 frames -> cannot fit in any 5-frame window
        obs = [
            ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=5, timestamp_ms=1000, x1=0, y1=0, x2=10, y2=10, conf=0.85),
        ]
        self.assertFalse(check_temporal_window(obs, min_obs=2, window_size_frames=5))

    def test_widely_separated_observations_fail(self):
        # Frame 0, 10, 20: each gap is 10 > 4
        obs = [
            ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=10, timestamp_ms=2000, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=20, timestamp_ms=4000, x1=0, y1=0, x2=10, y2=10, conf=0.8),
        ]
        self.assertFalse(check_temporal_window(obs, min_obs=2, window_size_frames=5))

    def test_burst_within_isolated_track_passes(self):
        # Hits at frame 0 (isolated), then frame 10 and 13 (gap 3 <= 4)
        obs = [
            ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=10, timestamp_ms=2000, x1=0, y1=0, x2=10, y2=10, conf=0.8),
            ObservedDetection(frame_index=13, timestamp_ms=2600, x1=0, y1=0, x2=10, y2=10, conf=0.85),
        ]
        self.assertTrue(check_temporal_window(obs, min_obs=2, window_size_frames=5))

    def test_tracker_predicted_boxes_below_conf_are_excluded(self):
        # Actual hit at frame 0 (conf 0.6), tracker prediction at frame 2 (conf 0.2 < threshold 0.35)
        obs = [
            ObservedDetection(frame_index=0, timestamp_ms=0, x1=0, y1=0, x2=10, y2=10, conf=0.6),
            ObservedDetection(frame_index=2, timestamp_ms=400, x1=0, y1=0, x2=10, y2=10, conf=0.2),
        ]
        self.assertFalse(check_temporal_window(obs, min_obs=2, window_size_frames=5, conf_threshold=0.35))


class TestPolicyApplicationAndIndependence(unittest.TestCase):
    """Tests for Policy A, Policy B, and Policy C independence."""

    def test_policy_a_confidence_gating(self):
        cfg = PolicyConfig(conf_threshold=0.35)
        ev_pass = make_test_event("ev1", conf_max=0.45, observations=[(0, 0, 0.45)])
        ev_fail = make_test_event("ev2", conf_max=0.25, observations=[(0, 0, 0.25)])

        events = apply_policies([ev_pass, ev_fail], cfg)
        self.assertTrue(events[0].policy_a_retained)
        self.assertFalse(events[1].policy_a_retained)

    def test_policy_c_independent_from_policy_b(self):
        """CRITICAL: Policy C must retain high-confidence visual verifications even when Policy B fails."""
        cfg = PolicyConfig(conf_threshold=0.35, verifier_threshold=0.50, policy_c_requires_policy_b=False)

        # Event has only 1 observation at frame 0 (fails Policy B), but strong verifier score 0.85
        ev = make_test_event("ev_single_hit", conf_max=0.60, observations=[(0, 0, 0.60)], policy_c_score=0.85)

        apply_policies([ev], cfg)
        self.assertTrue(ev.policy_a_retained)
        self.assertFalse(ev.policy_b_retained)   # Fails temporal rule (only 1 hit)
        self.assertTrue(ev.policy_c_retained)    # PASSES Policy C independently!

    def test_policy_c_when_no_checkpoint_loaded_is_none(self):
        """Policy C must remain None (unavailable) when no verifier is configured — never fabricated."""
        cfg = PolicyConfig()
        ev = make_test_event("ev_no_verifier", conf_max=0.60, observations=[(0, 0, 0.60)], policy_c_score=None)

        apply_policies([ev], cfg)
        self.assertIsNone(ev.policy_c_retained)

    def test_policy_c_explicit_combination_experiment(self):
        """When policy_c_requires_policy_b=True, both conditions are required."""
        cfg = PolicyConfig(conf_threshold=0.35, verifier_threshold=0.50, policy_c_requires_policy_b=True)

        ev_b_fail = make_test_event("ev1", conf_max=0.60, observations=[(0, 0, 0.60)], policy_c_score=0.90)
        ev_both_pass = make_test_event("ev2", conf_max=0.60, observations=[(0, 0, 0.60), (2, 400, 0.65)], policy_c_score=0.90)

        apply_policies([ev_b_fail, ev_both_pass], cfg)
        self.assertFalse(ev_b_fail.policy_c_retained)
        self.assertTrue(ev_both_pass.policy_c_retained)


if __name__ == "__main__":
    unittest.main()
