"""
Unit tests for Smart Leaf ROI Extractor & Aspect-Ratio Letterbox Preprocessor.
"""
import unittest
import cv2
import numpy as np

from src.inference.preprocessor import detect_leaf_roi, letterbox_image, preprocess_leaf
from src.inference.pipeline import LeafDocPipeline


class TestPreprocessor(unittest.TestCase):
    """Test suite for leaf ROI detection and letterboxing."""

    def test_01_detect_leaf_roi_on_outdoor_framed_leaf(self):
        """Verify detect_leaf_roi isolates the leaf blade when surrounded by branches/lawn borders."""
        # Create 400x600 synthetic image with gray/brown borders and green leaf in center
        img = np.full((400, 600, 3), fill_value=[60, 45, 30], dtype=np.uint8)  # brown branch background
        
        # Draw green leaf ellipse in center
        cv2.ellipse(img, (300, 200), (120, 70), 25, 0, 360, (35, 140, 45), -1)

        cropped, (bx, by, bw, bh), is_cropped = detect_leaf_roi(img, pad_pct=0.08)
        self.assertTrue(is_cropped)
        self.assertLess(bw, 600)
        self.assertLess(bh, 400)
        self.assertGreater(bw, 150)
        self.assertGreater(bh, 100)
        self.assertEqual(cropped.shape[0], bh)
        self.assertEqual(cropped.shape[1], bw)

    def test_02_detect_leaf_roi_skips_tight_leaf(self):
        """Verify detect_leaf_roi does not crop already tightly framed PlantVillage leaves."""
        # Create full green leaf image filling 90% of frame
        img = np.full((256, 256, 3), fill_value=[35, 145, 45], dtype=np.uint8)
        _, _, is_cropped = detect_leaf_roi(img)
        self.assertFalse(is_cropped)

    def test_03_letterbox_preserves_aspect_ratio(self):
        """Verify letterbox_image scales rectangular image without stretching and pads symmetrically."""
        # Wide 16:9 rectangular image: 160 x 90
        img = np.zeros((90, 160, 3), dtype=np.uint8)
        img[:, :] = (50, 120, 50)

        target_size = 224
        letterboxed = letterbox_image(img, target_size=target_size, pad_color=(128, 128, 128))

        self.assertEqual(letterboxed.shape, (target_size, target_size, 3))
        # Top and bottom should have gray padding (128, 128, 128)
        self.assertTrue(np.all(letterboxed[0, 112] == [128, 128, 128]))
        self.assertTrue(np.all(letterboxed[223, 112] == [128, 128, 128]))
        # Center should contain the green leaf color
        self.assertTrue(np.all(letterboxed[112, 112] == [50, 120, 50]))

    def test_04_preprocess_leaf_unified(self):
        """Verify preprocess_leaf returns both tensor-ready square and cropped leaf."""
        img = np.full((300, 500, 3), fill_value=[40, 40, 40], dtype=np.uint8)
        cv2.circle(img, (250, 150), 60, (40, 160, 45), -1)

        letterboxed, cropped, bbox, is_cropped = preprocess_leaf(img, auto_crop=True, target_size=224)
        self.assertEqual(letterboxed.shape, (224, 224, 3))
        self.assertTrue(is_cropped)
        self.assertEqual(len(bbox), 4)

    def test_05_pipeline_auto_crop_integration(self):
        """Verify pipeline diagnosis returns is_cropped and roi_bbox."""
        pipeline = LeafDocPipeline()
        img = np.full((300, 450, 3), fill_value=[70, 50, 30], dtype=np.uint8)
        cv2.circle(img, (225, 150), 70, (40, 150, 45), -1)

        res = pipeline.diagnose(img, auto_crop=True, generate_visualization=True)
        self.assertIn("is_cropped", res)
        self.assertIn("roi_bbox", res)
        self.assertIn("cropped_leaf", res["visualizations"])


if __name__ == "__main__":
    unittest.main()
