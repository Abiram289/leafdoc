import tempfile
import unittest
from pathlib import Path

from src.train.report import generate_reports


class TestReports(unittest.TestCase):

    def test_generate_reports(self):
        """Verify that generate_reports creates classification_report.txt, confusion_matrix.png, and CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            idx_to_class = {0: "Healthy", 1: "Early_Blight", 2: "Late_Blight"}
            labels = [0, 1, 2, 0, 1, 2, 0, 1]
            preds =  [0, 1, 2, 0, 2, 2, 0, 1]

            generate_reports(labels, preds, idx_to_class, out_dir)

            txt_report = out_dir / "stage2_classification_report.txt"
            png_cm = out_dir / "stage2_confusion_matrix.png"
            csv_cm = out_dir / "stage2_confusion_matrix.csv"

            self.assertTrue(txt_report.exists(), "stage2_classification_report.txt was not generated")
            self.assertTrue(png_cm.exists(), "stage2_confusion_matrix.png was not generated")
            self.assertTrue(csv_cm.exists(), "stage2_confusion_matrix.csv was not generated")

            # Check content of txt report
            txt_content = txt_report.read_text()
            self.assertIn("Healthy", txt_content)
            self.assertIn("Early_Blight", txt_content)
            self.assertIn("Late_Blight", txt_content)

            # Check content of csv
            csv_content = csv_cm.read_text()
            self.assertIn("Healthy,Early_Blight,Late_Blight", csv_content)


if __name__ == "__main__":
    unittest.main()
