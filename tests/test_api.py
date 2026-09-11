"""
Unit and Integration Tests for LeafDoc FastAPI REST API & Serving Endpoints.
"""
import io
import unittest
from PIL import Image
from starlette.testclient import TestClient

from app.main import app


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints."""

    @classmethod
    def setUpClass(cls):
        # Initialize TestClient within lifespan context
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        """Verify /api/v1/health returns ok status and hardware/model metadata."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("device", data)
        self.assertEqual(data["total_classes"], 38)
        self.assertGreaterEqual(data["total_treatments"], 38)
        self.assertIn("EfficientNet-B0", data["stage1_checkpoint"])
        self.assertIn("EfficientNet-B0", data["stage2_checkpoint"])

    def test_02_classes_endpoint(self):
        """Verify /api/v1/classes returns all 38 fine-grained classes with species."""
        response = self.client.get("/api/v1/classes")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_classes"], 38)
        self.assertEqual(len(data["classes"]), 38)
        self.assertGreaterEqual(len(data["species_list"]), 10)
        
        # Check first item schema
        item = data["classes"][0]
        self.assertIn("fine_label", item)
        self.assertIn("species", item)
        self.assertIn("condition", item)
        self.assertIn("is_healthy", item)

    def test_03_treatments_endpoints(self):
        """Verify /api/v1/treatments and /api/v1/treatments/{fine_label} endpoints."""
        # 1. All treatments
        response = self.client.get("/api/v1/treatments")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 38)

        # 2. Species filter
        response = self.client.get("/api/v1/treatments?species=Tomato")
        self.assertEqual(response.status_code, 200)
        t_data = response.json()
        self.assertGreaterEqual(t_data["total"], 8)

        # 3. Specific label
        response = self.client.get("/api/v1/treatments/Apple___Apple_scab")
        self.assertEqual(response.status_code, 200)
        rec = response.json()
        self.assertEqual(rec["species"], "Apple")
        self.assertEqual(rec["condition_name"], "Apple Scab")
        self.assertIn("cultural_controls", rec)
        self.assertIn("immediate_action", rec)

    def test_04_samples_endpoint(self):
        """Verify /api/v1/samples lists available sample leaf items."""
        response = self.client.get("/api/v1/samples")
        self.assertEqual(response.status_code, 200)
        samples = response.json()
        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("id", s)
            self.assertIn("title", s)
            self.assertIn("url", s)

    def test_05_diagnose_multipart_upload(self):
        """Verify /api/v1/diagnose with uploaded image file returns full 4 stages and visual base64 data."""
        # Create a synthetic 224x224 RGB image in memory
        img = Image.new("RGB", (224, 224), color=(60, 150, 40))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        response = self.client.post(
            "/api/v1/diagnose",
            files={"file": ("test_leaf.jpg", img_bytes, "image/jpeg")},
            data={"include_visualizations": "true", "top_k": "5"},
        )
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertTrue(res["success"])
        self.assertIn("is_healthy", res)
        self.assertIn("overall_status", res)
        
        # Verify stage 1
        s1 = res["stage1_binary"]
        self.assertIn("label", s1)
        self.assertIn("confidence", s1)
        self.assertIn("probabilities", s1)

        # Verify stage 2
        s2 = res["stage2_fine"]
        self.assertIn("fine_label", s2)
        self.assertIn("species", s2)
        self.assertEqual(len(s2["top_predictions"]), 5)

        # Verify stage 3
        s3 = res["stage3_severity"]
        self.assertIn("severity", s3)
        self.assertIn("affected_area_pct", s3)

        # Verify stage 4
        s4 = res["stage4_treatment"]
        self.assertIn("immediate_action", s4)
        self.assertIn("cultural_controls", s4)

        # Verify visualizations
        vis = res["visualizations"]
        self.assertIsNotNone(vis)
        self.assertTrue(vis["cam_overlay"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(vis["composite_panel"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(vis["disease_mask"].startswith("data:image/png;base64,"))

    def test_06_diagnose_missing_input_raises_400(self):
        """Verify /api/v1/diagnose returns 400 Bad Request if neither file nor URL is given."""
        response = self.client.post("/api/v1/diagnose", data={})
        self.assertEqual(response.status_code, 400)

    def test_07_root_dashboard_html(self):
        """Verify root route '/' serves the HTML dashboard."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("LeafDoc", response.text)
        self.assertIn("Explainable Agronomic AI", response.text)


if __name__ == "__main__":
    unittest.main()
