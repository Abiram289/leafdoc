"""
Unit and Integration Tests for Two-Stage Neural Leaf Detection (YOLOv8).
Tests LeafDetector initialization, leaf bounding box detection, candidate ranking,
detection box rendering, pipeline integration, and graceful fallback behavior.
"""
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from starlette.testclient import TestClient

from app.main import app
from src.inference.detector import LeafDetector
from src.inference.pipeline import LeafDocPipeline


class TestLeafDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = LeafDetector()
        cls.sample_path = Path("app/static/samples/apple_scab.jpg")
        if cls.sample_path.exists():
            cls.sample_rgb = cv2.cvtColor(cv2.imread(str(cls.sample_path)), cv2.COLOR_BGR2RGB)
        else:
            cls.sample_rgb = np.full((256, 256, 3), 128, dtype=np.uint8)

    def test_01_detector_initialization(self):
        """Verify LeafDetector initializes with local weights or downloads successfully."""
        self.assertTrue(
            self.detector.is_available(),
            "LeafDetector model should be loaded and available.",
        )
        self.assertIsNotNone(self.detector.model)
        self.assertIsInstance(self.detector.names, dict)
        self.assertGreater(len(self.detector.names), 10)

    def test_02_detect_leaf_bboxes(self):
        """Verify detect returns structured candidate leaf items with valid bbox coordinates."""
        if not self.sample_path.exists():
            self.skipTest("Sample image app/static/samples/apple_scab.jpg not found.")

        leaves = self.detector.detect(self.sample_rgb)
        self.assertIsInstance(leaves, list)
        self.assertGreaterEqual(len(leaves), 1, "Should detect at least 1 leaf in apple_scab.jpg")

        h, w = self.sample_rgb.shape[:2]
        for leaf in leaves:
            self.assertIn("index", leaf)
            self.assertIn("bbox", leaf)
            self.assertIn("confidence", leaf)
            self.assertIn("label", leaf)
            self.assertIn("area", leaf)
            self.assertIn("is_primary", leaf)

            x1, y1, x2, y2 = leaf["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)
            self.assertGreater(leaf["confidence"], 0.0)

        # Primary leaf should be index 0
        self.assertTrue(leaves[0]["is_primary"])

    def test_03_select_primary_leaf_ranking_and_override(self):
        """Verify candidate selection chooses primary by default, or respects selected_box_idx."""
        mock_leaves = [
            {
                "index": 0,
                "bbox": [10, 10, 100, 100],
                "confidence": 0.85,
                "class_id": 28,
                "label": "apple",
                "area": 8100,
                "is_primary": True,
            },
            {
                "index": 1,
                "bbox": [120, 120, 200, 200],
                "confidence": 0.70,
                "class_id": 26,
                "label": "cherry",
                "area": 6400,
                "is_primary": False,
            },
        ]

        # Default selection -> index 0
        primary_default = self.detector.select_primary_leaf(
            mock_leaves,
            image_shape=(256, 256),
        )
        self.assertEqual(primary_default["index"], 0)
        self.assertTrue(primary_default["is_primary"])

        # User override -> index 1
        primary_override = self.detector.select_primary_leaf(
            mock_leaves,
            image_shape=(256, 256),
            selected_box_idx=1,
        )
        self.assertEqual(primary_override["index"], 1)
        self.assertTrue(primary_override["is_primary"])
        self.assertFalse(mock_leaves[0]["is_primary"])

    def test_04_draw_detection_overlay(self):
        """Verify draw_detection_boxes produces an annotated RGB array of identical dimensions."""
        mock_leaves = [
            {
                "index": 0,
                "bbox": [20, 20, 120, 120],
                "confidence": 0.92,
                "class_id": 28,
                "label": "apple",
                "area": 10000,
                "is_primary": True,
            }
        ]
        overlay = self.detector.draw_detection_boxes(self.sample_rgb, mock_leaves)
        self.assertIsInstance(overlay, np.ndarray)
        self.assertEqual(overlay.shape, self.sample_rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)

    def test_05_crop_leaf_roi(self):
        """Verify crop_leaf_roi crops image preserving padding margin."""
        bbox = [50, 50, 150, 150]
        cropped, (rx, ry, rw, rh) = self.detector.crop_leaf_roi(self.sample_rgb, bbox, pad_pct=0.05)
        self.assertIsInstance(cropped, np.ndarray)
        self.assertEqual(cropped.shape[0], rh)
        self.assertEqual(cropped.shape[1], rw)
        self.assertEqual(cropped.shape[2], 3)
        self.assertGreater(rw, 0)
        self.assertGreater(rh, 0)

    def test_06_pipeline_yolo_diagnose_integration(self):
        """Verify LeafDocPipeline diagnose runs with YOLO detector and returns expected fields."""
        if not self.sample_path.exists():
            self.skipTest("Sample image not found.")

        pipeline = LeafDocPipeline()
        result = pipeline.diagnose(
            image_input=str(self.sample_path),
            use_neural_detector=True,
            generate_visualization=True,
        )

        self.assertIn("detector_used", result)
        self.assertEqual(result["detector_used"], "yolov8")
        self.assertIn("detected_leaves", result)
        self.assertGreaterEqual(len(result["detected_leaves"]), 1)
        self.assertIn("detection_overlay", result["visualizations"])
        self.assertIsNotNone(result["visualizations"]["detection_overlay"])
        self.assertIn("stage2_fine", result)
        self.assertEqual(result["stage2_fine"]["species"], "Apple")

    def test_07_pipeline_fallback_when_yolo_disabled(self):
        """Verify pipeline falls back to GrabCut preprocessor when use_neural_detector=False."""
        pipeline = LeafDocPipeline()
        result = pipeline.diagnose(
            image_input=str(self.sample_path),
            use_neural_detector=False,
            auto_crop=True,
        )
        self.assertIn(result["detector_used"], ("grabcut", "none"))

    def test_08_api_diagnose_endpoint_with_yolo(self):
        """Verify POST /api/v1/diagnose returns detector_used, detected_leaves, and detection_overlay."""
        if not self.sample_path.exists():
            self.skipTest("Sample image not found.")

        client = TestClient(app)
        with open(self.sample_path, "rb") as f:
            resp = client.post(
                "/api/v1/diagnose",
                files={"file": ("apple_scab.jpg", f, "image/jpeg")},
                data={"use_neural_detector": "true"},
            )

        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["detector_used"], "yolov8")
        self.assertGreaterEqual(len(payload["detected_leaves"]), 1)
        self.assertIn("detection_overlay", payload["visualizations"])
        self.assertIsNotNone(payload["visualizations"]["detection_overlay"])
        self.assertTrue(payload["visualizations"]["detection_overlay"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
