"""
Unit tests for Stage 3: Grad-CAM Explainability, Leaf Masking, and Severity Estimation.
"""
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.inference.gradcam import (
    GradCAMExplainer,
    compute_severity,
    generate_annotated_panel,
    generate_gradcam_overlay,
    get_target_layers,
    segment_leaf_mask,
)
from src.models.factory import build_model


class TestGradCAMAndSeverity(unittest.TestCase):

    def setUp(self):
        # Instantiate small test model
        self.model = build_model("efficientnet_b0", num_classes=38, pretrained=False)
        self.model.eval()

    def test_target_layers_efficientnet(self):
        """Verify get_target_layers finds conv_head on EfficientNet."""
        layers = get_target_layers(self.model, "efficientnet_b0")
        self.assertEqual(len(layers), 1)
        self.assertTrue(isinstance(layers[0], nn.Module))

    def test_gradcam_heatmap_generation(self):
        """Verify GradCAMExplainer generates valid 2D float32 heatmap in [0, 1]."""
        explainer = GradCAMExplainer(self.model, backbone_name="efficientnet_b0", device=torch.device("cpu"))
        dummy_tensor = torch.randn(1, 3, 224, 224)
        heatmap = explainer.generate_heatmap(dummy_tensor, target_class_idx=5)

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(np.all(heatmap >= 0.0))
        self.assertTrue(np.all(heatmap <= 1.0))
        self.assertEqual(heatmap.dtype, np.float32)

    def test_leaf_mask_segmentation(self):
        """Verify leaf mask segmentation creates a binary mask."""
        # Create a synthetic image with a green circle on a gray background
        h, w = 224, 224
        img = np.full((h, w, 3), 128, dtype=np.uint8)  # Gray background
        # Green patch
        img[50:180, 50:180] = [34, 139, 34]  # Forest green

        mask = segment_leaf_mask(img)
        self.assertEqual(mask.shape, (h, w))
        self.assertTrue(np.all((mask == 0) | (mask == 255)))
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_compute_severity_thresholds(self):
        """Verify severity categories match thresholds: <15% Mild, 15-40% Moderate, >40% Severe."""
        h, w = 100, 100
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255  # 10,000 total pixels

        # 1. Healthy test
        res_healthy = compute_severity(np.ones((h, w), dtype=np.float32), leaf_mask, is_healthy=True)
        self.assertEqual(res_healthy["severity"], "Healthy")
        self.assertEqual(res_healthy["affected_area_pct"], 0.0)

        # 2. Mild test (< 15% active)
        cam_mild = np.zeros((h, w), dtype=np.float32)
        cam_mild[:10, :10] = 0.9  # 100 / 10000 = 1%
        res_mild = compute_severity(cam_mild, leaf_mask, is_healthy=False, activation_threshold=0.5, mild_max_pct=15.0, moderate_max_pct=40.0)
        self.assertEqual(res_mild["severity"], "Mild")
        self.assertAlmostEqual(res_mild["affected_area_pct"], 1.0, places=1)

        # 3. Moderate test (15-40%)
        cam_mod = np.zeros((h, w), dtype=np.float32)
        cam_mod[:25, :100] = 0.9  # 2500 / 10000 = 25%
        res_mod = compute_severity(cam_mod, leaf_mask, is_healthy=False, activation_threshold=0.5, mild_max_pct=15.0, moderate_max_pct=40.0)
        self.assertEqual(res_mod["severity"], "Moderate")
        self.assertAlmostEqual(res_mod["affected_area_pct"], 25.0, places=1)

        # 4. Severe test (> 40%)
        cam_sev = np.zeros((h, w), dtype=np.float32)
        cam_sev[:50, :100] = 0.9  # 5000 / 10000 = 50%
        res_sev = compute_severity(cam_sev, leaf_mask, is_healthy=False, activation_threshold=0.5, mild_max_pct=15.0, moderate_max_pct=40.0)
        self.assertEqual(res_sev["severity"], "Severe")
        self.assertAlmostEqual(res_sev["affected_area_pct"], 50.0, places=1)

    def test_overlay_and_panel_generation(self):
        """Verify Grad-CAM overlay and 4-panel dashboard generate valid RGB arrays."""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        overlay = generate_gradcam_overlay(img, heatmap)

        self.assertEqual(overlay.shape, (224, 224, 3))
        self.assertEqual(overlay.dtype, np.uint8)

        disease_mask = (heatmap > 0.5).astype(np.uint8) * 255
        panel = generate_annotated_panel(
            image_rgb=img,
            cam_overlay=overlay,
            disease_mask=disease_mask,
            species="Tomato",
            condition_name="Early Blight",
            confidence=0.985,
            severity="Moderate",
            affected_pct=25.4,
            target_size=(256, 256),
        )

        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[0], 256)
        self.assertEqual(panel.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
