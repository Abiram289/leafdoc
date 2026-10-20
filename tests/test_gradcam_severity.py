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

    def test_leaf_mask_with_colored_background(self):
        """Verify segment_leaf_mask isolates green leaf even on lavender/purple paper backdrops."""
        h, w = 224, 224
        # Lavender/purple backdrop common in PlantVillage Apple Scab images
        img = np.full((h, w, 3), [180, 160, 205], dtype=np.uint8)
        # Green leaf blade
        img[40:180, 40:180] = [60, 140, 50]
        mask = segment_leaf_mask(img)
        self.assertEqual(mask.shape, (h, w))
        # Mask should identify the green leaf area, rejecting lavender backdrop
        self.assertGreater(mask[100, 100], 0)
        self.assertEqual(mask[10, 10], 0)

    def test_compute_severity_excludes_healthy_green_tissue(self):
        """Verify compute_severity does not mark pristine healthy green foliage as diseased lesions."""
        h, w = 100, 100
        # Synthetic leaf: 95% vibrant green, 5% brown spot
        img = np.full((h, w, 3), [40, 150, 40], dtype=np.uint8)  # healthy green
        img[10:15, 10:15] = [120, 60, 20]  # small brown lesion (25 pixels = 0.25%)
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Diffuse high Grad-CAM activation covering the whole leaf (e.g. 0.8)
        cam = np.full((h, w), 0.8, dtype=np.float32)

        res = compute_severity(cam, leaf_mask, is_healthy=False, image_rgb=img)
        # Affected area should reflect physical lesions, NOT 100% of the green leaf
        self.assertLess(res["affected_area_pct"], 15.0)
        self.assertEqual(res["severity"], "Mild")

    def test_overlay_adaptive_transparency_and_mask(self):
        """Verify generate_gradcam_overlay leaves background and sub-threshold healthy regions natural."""
        h, w = 100, 100
        bg_color = np.array([200, 200, 200], dtype=np.uint8)
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        # Leaf in center
        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        leaf_mask[20:80, 20:80] = 255
        img[20:80, 20:80] = [50, 160, 50]

        # Cam heatmap with peak inside leaf and 0 outside
        cam = np.zeros((h, w), dtype=np.float32)
        cam[30:40, 30:40] = 0.9  # Hotspot
        cam[50:60, 50:60] = 0.05  # Sub-threshold low activation

        overlay = generate_gradcam_overlay(img, cam, leaf_mask=leaf_mask, activation_floor=0.18)

        # Background pixel (5, 5) outside leaf should be identical to original image
        np.testing.assert_array_equal(overlay[5, 5], img[5, 5])
        # Sub-threshold activation (55, 55) inside leaf should be untouched original leaf green
        np.testing.assert_array_equal(overlay[55, 55], img[55, 55])
        # Hotspot pixel (35, 35) should be modified by the colormap
        self.assertFalse(np.array_equal(overlay[35, 35], img[35, 35]))


if __name__ == "__main__":
    unittest.main()
