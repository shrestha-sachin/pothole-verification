"""Test evidence observation selection among top 3, tie breaking, 15% context clamping, and ImageNet normalization."""

import unittest
import numpy as np
from src.tracking.event_grouper import ObservedDetection, select_representative_observation
from src.verifier.preprocessing import (
    DEFAULT_PAD_FRAC,
    IMAGENET_MEAN_RGB,
    IMAGENET_STD_RGB,
    TARGET_SIZE,
    expand_and_clamp_box,
    extract_padded_crop,
    preprocess_crop_for_verifier,
)


class TestEvidenceSelection(unittest.TestCase):
    """Tests for selecting the representative observation from the top 3 confidence candidates."""

    def test_sharpest_selected_strictly_from_top_three_confidence(self):
        """CRITICAL: An observation with high sharpness outside the top 3 confidence must NOT be selected."""
        # 5 observations:
        # Obs 1: conf 0.95, lap 50   (in top 3)
        # Obs 2: conf 0.90, lap 200  (in top 3) -> HIGHEST SHARPNESS IN TOP 3
        # Obs 3: conf 0.85, lap 100  (in top 3)
        # Obs 4: conf 0.70, lap 999  (outside top 3: highest overall sharpness)
        # Obs 5: conf 0.50, lap 1500 (outside top 3)
        obs_list = [
            ObservedDetection(frame_index=1, timestamp_ms=200, x1=10, y1=10, x2=50, y2=50, conf=0.95, laplacian_var=50.0),
            ObservedDetection(frame_index=2, timestamp_ms=400, x1=10, y1=10, x2=50, y2=50, conf=0.90, laplacian_var=200.0),
            ObservedDetection(frame_index=3, timestamp_ms=600, x1=10, y1=10, x2=50, y2=50, conf=0.85, laplacian_var=100.0),
            ObservedDetection(frame_index=4, timestamp_ms=800, x1=10, y1=10, x2=50, y2=50, conf=0.70, laplacian_var=999.0),
            ObservedDetection(frame_index=5, timestamp_ms=1000, x1=10, y1=10, x2=50, y2=50, conf=0.50, laplacian_var=1500.0),
        ]

        best = select_representative_observation(obs_list)
        self.assertIsNotNone(best)
        # Must select Obs 2 (conf 0.90, lap 200.0), NOT Obs 4 or 5!
        self.assertEqual(best.frame_index, 2)
        self.assertEqual(best.conf, 0.90)
        self.assertEqual(best.laplacian_var, 200.0)

    def test_tie_breaking_by_confidence(self):
        """When Laplacian variances are identical in top 3, break tie by higher confidence."""
        obs_list = [
            ObservedDetection(frame_index=1, timestamp_ms=200, x1=10, y1=10, x2=50, y2=50, conf=0.85, laplacian_var=150.0),
            ObservedDetection(frame_index=2, timestamp_ms=400, x1=10, y1=10, x2=50, y2=50, conf=0.92, laplacian_var=150.0),
            ObservedDetection(frame_index=3, timestamp_ms=600, x1=10, y1=10, x2=50, y2=50, conf=0.80, laplacian_var=150.0),
        ]

        best = select_representative_observation(obs_list)
        self.assertIsNotNone(best)
        # Obs 2 has higher confidence (0.92 vs 0.85)
        self.assertEqual(best.frame_index, 2)
        self.assertEqual(best.conf, 0.92)


class TestCropAndPreprocessing(unittest.TestCase):
    """Tests for 15% context expansion, border clamping, aspect ratio preservation, and normalization."""

    def test_fifteen_percent_context_expansion_interior(self):
        # 100x100 box interior to 1000x1000 image
        # x1=100, y1=100, x2=200, y2=200 -> w=100, h=100
        # 15% of 100 = 15 px padding on each side
        cx1, cy1, cx2, cy2 = expand_and_clamp_box(100, 100, 200, 200, img_w=1000, img_h=1000, pad_frac=0.15)
        self.assertEqual(cx1, 85)
        self.assertEqual(cy1, 85)
        self.assertEqual(cx2, 215)
        self.assertEqual(cy2, 215)

    def test_context_expansion_clamped_at_borders(self):
        # Box near top-left corner
        cx1, cy1, cx2, cy2 = expand_and_clamp_box(5, 8, 105, 108, img_w=1000, img_h=1000, pad_frac=0.15)
        self.assertEqual(cx1, 0)   # Clamped at 0 (5 - 15 = -10 -> 0)
        self.assertEqual(cy1, 0)   # Clamped at 0 (8 - 15 = -7 -> 0)
        self.assertEqual(cx2, 120)
        self.assertEqual(cy2, 123)

        # Box near bottom-right corner
        cx1, cy1, cx2, cy2 = expand_and_clamp_box(950, 950, 995, 995, img_w=1000, img_h=1000, pad_frac=0.15)
        self.assertEqual(cx2, 1000)  # Clamped at img_w
        self.assertEqual(cy2, 1000)  # Clamped at img_h

    def test_aspect_ratio_preservation_and_padding_to_224(self):
        # Create non-square test crop (400 wide x 200 high)
        crop_wide = np.zeros((200, 400, 3), dtype=np.uint8)
        tensor = preprocess_crop_for_verifier(crop_wide, target_size=TARGET_SIZE)

        self.assertEqual(tensor.shape, (3, TARGET_SIZE, TARGET_SIZE))
        self.assertEqual(tensor.dtype, np.float32)

        # Padded top/bottom borders should match normalized ImageNet mean
        # When canvas is filled with IMAGENET_MEAN_RGB, normalized value is (mean - mean)/std = 0.0
        np.testing.assert_allclose(tensor[:, 0, 0], [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(tensor[:, TARGET_SIZE - 1, 0], [0.0, 0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
