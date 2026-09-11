"""
Unit tests for Agronomic Treatments Encyclopedia and Recommendation Engine.
"""
import json
from pathlib import Path
import unittest

from src.inference.treatments import TreatmentRecommender


class TestTreatments(unittest.TestCase):

    def setUp(self):
        self.recommender = TreatmentRecommender()
        self.fine_map_path = Path("data/splits/class_map_fine.json")

    def test_all_38_classes_covered(self):
        """Verify every single class in class_map_fine.json exists in treatments.json."""
        if not self.fine_map_path.exists():
            self.skipTest("class_map_fine.json not present.")

        with open(self.fine_map_path, "r", encoding="utf-8") as f:
            fine_map = json.load(f)

        self.assertEqual(len(fine_map), 38)
        for class_name in fine_map.keys():
            self.assertIn(
                class_name,
                self.recommender.database,
                f"Class '{class_name}' missing from treatments.json",
            )

    def test_entry_schema(self):
        """Verify each treatment record has all required fields and non-empty controls."""
        required_keys = [
            "species",
            "condition_name",
            "pathogen_type",
            "description",
            "cultural_controls",
            "chemical_controls",
            "biological_controls",
            "preventive_measures",
            "severity_actions",
        ]

        for label, data in self.recommender.database.items():
            for key in required_keys:
                self.assertIn(key, data, f"Missing '{key}' in record for {label}")

            self.assertIsInstance(data["cultural_controls"], list)
            self.assertGreater(len(data["cultural_controls"]), 0)
            self.assertIsInstance(data["preventive_measures"], list)
            self.assertGreater(len(data["preventive_measures"]), 0)
            self.assertIsInstance(data["severity_actions"], dict)

    def test_get_recommendation_diseased_and_healthy(self):
        """Verify get_recommendation returns appropriate action for various severities."""
        # Diseased test
        rec_sev = self.recommender.get_recommendation("Tomato___Early_blight", severity="Severe")
        self.assertEqual(rec_sev["species"], "Tomato")
        self.assertEqual(rec_sev["severity_level"], "Severe")
        self.assertIn("fungicide", rec_sev["immediate_action"].lower())

        # Healthy test
        rec_healthy = self.recommender.get_recommendation("Tomato___healthy", severity="Healthy")
        self.assertEqual(rec_healthy["species"], "Tomato")
        self.assertEqual(rec_healthy["severity_level"], "Healthy")

    def test_format_text_summary(self):
        """Verify format_text_summary produces readable multi-line text."""
        summary = self.recommender.format_text_summary("Apple___Black_rot", severity="Moderate")
        self.assertIn("Black Rot", summary)
        self.assertIn("Immediate Action Plan [Moderate]:", summary)
        self.assertIn("Cultural Management:", summary)


if __name__ == "__main__":
    unittest.main()
