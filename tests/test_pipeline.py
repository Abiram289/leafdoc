"""
Integration tests for LeafDocPipeline (End-to-End multi-stage diagnosis).
"""
import unittest
from pathlib import Path
import numpy as np

from src.inference.pipeline import LeafDocPipeline


class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stage1_path = Path("checkpoints/stage1_binary_best.pth")
        cls.stage2_path = Path("checkpoints/stage2_fine_best.pth")

        if not (cls.stage1_path.exists() and cls.stage2_path.exists()):
            cls.pipeline = None
        else:
            cls.pipeline = LeafDocPipeline(
                config_path="configs/config.yaml",
                stage1_ckpt_path=str(cls.stage1_path),
                stage2_ckpt_path=str(cls.stage2_path),
            )

    def test_pipeline_initialization(self):
        """Verify pipeline loads models, explainer, and treatments properly."""
        if self.pipeline is None:
            self.skipTest("Checkpoints not available.")

        self.assertIsNotNone(self.pipeline.s1_model)
        self.assertIsNotNone(self.pipeline.s2_model)
        self.assertIsNotNone(self.pipeline.explainer)
        self.assertIsNotNone(self.pipeline.recommender)

    def test_pipeline_diagnose_numpy_array(self):
        """Verify end-to-end diagnosis runs on a synthetic RGB array and returns all stage keys."""
        if self.pipeline is None:
            self.skipTest("Checkpoints not available.")

        # Synthetic green leaf image with a small dark lesion
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        dummy_img[40:180, 40:180] = [34, 139, 34]     # Green leaf tissue
        dummy_img[80:110, 80:110] = [139, 69, 19]     # Brown lesion spot

        result = self.pipeline.diagnose(dummy_img, generate_visualization=True)

        self.assertIn("stage1_binary", result)
        self.assertIn("stage2_fine", result)
        self.assertIn("stage3_severity", result)
        self.assertIn("stage4_treatment", result)
        self.assertIn("visualizations", result)

        self.assertIn(result["overall_status"], ["Healthy", "Diseased"])
        self.assertIn(result["stage3_severity"]["severity"], ["Healthy", "Mild", "Moderate", "Severe"])
        self.assertIsInstance(result["stage3_severity"]["affected_area_pct"], float)

        # Verify visualization composite panel was generated
        panel = result["visualizations"]["composite_panel"]
        self.assertIsNotNone(panel)
        self.assertEqual(panel.ndim, 3)
        self.assertEqual(panel.shape[2], 3)

    def test_pipeline_target_species_filtering(self):
        """Verify target_species locks Stage 2 inference and returns only classes for that crop."""
        if self.pipeline is None:
            self.skipTest("Checkpoints not available.")

        # Test species list
        supported = self.pipeline.get_supported_species()
        self.assertGreaterEqual(len(supported), 14)
        self.assertIn("Potato", supported)
        self.assertIn("Tomato", supported)
        self.assertIn("Apple", supported)

        # Synthetic green leaf image
        dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
        dummy_img[40:180, 40:180] = [34, 139, 34]

        # 1. Constrained to Potato
        result_potato = self.pipeline.diagnose(dummy_img, generate_visualization=False, target_species="Potato")
        self.assertEqual(result_potato["target_species"], "Potato")
        self.assertEqual(result_potato["stage2_fine"]["species"], "Potato")
        for pred in result_potato["stage2_fine"]["top_predictions"]:
            self.assertTrue(pred["label"].startswith("Potato___"), f"Expected Potato class, got {pred['label']}")

        # 2. Constrained to Tomato
        result_tomato = self.pipeline.diagnose(dummy_img, generate_visualization=False, target_species="Tomato")
        self.assertEqual(result_tomato["target_species"], "Tomato")
        self.assertEqual(result_tomato["stage2_fine"]["species"], "Tomato")
        for pred in result_tomato["stage2_fine"]["top_predictions"]:
            self.assertTrue(pred["label"].startswith("Tomato___"), f"Expected Tomato class, got {pred['label']}")

        # 3. Unconstrained (target_species=None)
        result_unconstrained = self.pipeline.diagnose(dummy_img, generate_visualization=False, target_species=None)
        self.assertIsNone(result_unconstrained["target_species"])


if __name__ == "__main__":
    unittest.main()

