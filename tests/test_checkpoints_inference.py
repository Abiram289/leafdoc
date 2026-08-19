import unittest
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F

from src.data.augmentations import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config


class TestCheckpointsAndInference(unittest.TestCase):

    def test_stage1_checkpoint_and_inference(self):
        """Verify Stage 1 checkpoint integrity, weight loading, and live inference logic."""
        cfg = load_config("configs/config.yaml")
        ckpt_path = Path(cfg["checkpoints_dir"]) / cfg["stage1"]["checkpoint_name"]

        if not ckpt_path.exists():
            self.skipTest(f"Stage 1 checkpoint {ckpt_path} not found.")

        device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Checkpoint structure
        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("class_to_idx", checkpoint)
        self.assertIn("backbone", checkpoint)
        self.assertEqual(len(checkpoint["class_to_idx"]), 2)

        # Instantiate model and load state
        model = build_model(checkpoint["backbone"], num_classes=2, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Synthetic inference test
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = F.softmax(logits, dim=1)[0]

        self.assertEqual(probs.shape, (2,))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_stage2_checkpoint_and_inference(self):
        """Verify Stage 2 checkpoint integrity, weight loading, and live inference logic."""
        cfg = load_config("configs/config.yaml")
        ckpt_path = Path(cfg["checkpoints_dir"]) / cfg["stage2"]["checkpoint_name"]

        if not ckpt_path.exists():
            self.skipTest(f"Stage 2 checkpoint {ckpt_path} not found.")

        device = torch.device("cpu")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        self.assertIn("model_state_dict", checkpoint)
        self.assertIn("class_to_idx", checkpoint)
        self.assertEqual(len(checkpoint["class_to_idx"]), 38)

        model = build_model(checkpoint["backbone"], num_classes=38, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_input)
            probs = F.softmax(logits, dim=1)[0]

        self.assertEqual(probs.shape, (38,))
        self.assertAlmostEqual(probs.sum().item(), 1.0, places=4)

    def test_live_image_inference(self):
        """Verify end-to-end inference on a real leaf image if available."""
        cfg = load_config("configs/config.yaml")
        sample_img_path = Path(r"D:\PROJECT\test_online\image4.jpg")

        if not sample_img_path.exists():
            self.skipTest(f"Sample test image {sample_img_path} not found.")

        ckpt_path = Path(cfg["checkpoints_dir"]) / cfg["stage1"]["checkpoint_name"]
        if not ckpt_path.exists():
            self.skipTest(f"Stage 1 checkpoint not found.")

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        class_to_idx = checkpoint["class_to_idx"]

        model = build_model(checkpoint["backbone"], num_classes=len(class_to_idx), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Preprocessing
        image = cv2.imread(str(sample_img_path))
        self.assertIsNotNone(image, "Failed to load sample image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transforms = get_val_transforms(cfg)
        tensor = transforms(image=image)["image"].unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        self.assertGreater(probs.sum().item(), 0.99)
        self.assertEqual(probs.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
