import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.dataset import LeafDataset
from src.data.prepare_dataset import derive_labels
from src.utils.config import load_config


class TestDataPipeline(unittest.TestCase):

    def test_derive_labels(self):
        """Verify that folder names are correctly parsed into species, disease, and binary labels."""
        # Standard diseased folder
        species, disease, binary = derive_labels("Tomato___Late_blight")
        self.assertEqual(species, "Tomato")
        self.assertEqual(disease, "Late_blight")
        self.assertEqual(binary, "diseased")

        # Standard healthy folder
        species, disease, binary = derive_labels("Apple___healthy")
        self.assertEqual(species, "Apple")
        self.assertEqual(disease, "healthy")
        self.assertEqual(binary, "healthy")

        # Multiple underscores in disease name
        species, disease, binary = derive_labels("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")
        self.assertEqual(species, "Corn_(maize)")
        self.assertEqual(disease, "Cercospora_leaf_spot Gray_leaf_spot")
        self.assertEqual(binary, "diseased")

    def test_augmentations_output_shapes(self):
        """Verify Albumentations transform pipelines produce correctly sized normalized tensors."""
        cfg = load_config("configs/config.yaml")
        img_size = cfg["data"]["image_size"]

        dummy_image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

        train_tf = get_train_transforms(cfg)
        val_tf = get_val_transforms(cfg)

        train_out = train_tf(image=dummy_image)["image"]
        val_out = val_tf(image=dummy_image)["image"]

        self.assertIsInstance(train_out, torch.Tensor)
        self.assertIsInstance(val_out, torch.Tensor)
        self.assertEqual(train_out.shape, (3, img_size, img_size))
        self.assertEqual(val_out.shape, (3, img_size, img_size))
        self.assertEqual(train_out.dtype, torch.float32)

    def test_leaf_dataset_with_mock_data(self):
        """Verify LeafDataset loading, indexing, and persistent class mapping with mock CSV and images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img1_path = tmp_path / "img1.jpg"
            img2_path = tmp_path / "img2.jpg"

            # Create dummy image files
            cv2.imwrite(str(img1_path), np.full((100, 100, 3), 128, dtype=np.uint8))
            cv2.imwrite(str(img2_path), np.full((100, 100, 3), 200, dtype=np.uint8))

            csv_path = tmp_path / "mock_split.csv"
            df = pd.DataFrame([
                {"filepath": str(img1_path), "binary_label": "healthy", "fine_label": "Apple___healthy"},
                {"filepath": str(img2_path), "binary_label": "diseased", "fine_label": "Apple___Black_rot"},
            ])
            df.to_csv(csv_path, index=False)

            class_map_path = tmp_path / "class_map.json"

            cfg = load_config("configs/config.yaml")
            val_tf = get_val_transforms(cfg)

            dataset = LeafDataset(
                csv_path=str(csv_path),
                label_col="binary_label",
                transforms=val_tf,
                class_map_path=str(class_map_path),
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.num_classes, 2)
            self.assertTrue(class_map_path.exists(), "class_map_path should be created if not existing")

            img_tensor, label_idx = dataset[0]
            self.assertIsInstance(img_tensor, torch.Tensor)
            self.assertIsInstance(label_idx, int)
            self.assertTrue(0 <= label_idx < dataset.num_classes)


if __name__ == "__main__":
    unittest.main()
