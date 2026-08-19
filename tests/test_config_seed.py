import os
import random
import unittest
from pathlib import Path
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed


class TestConfigAndSeed(unittest.TestCase):

    def test_seed_determinism(self):
        """Verify that set_seed ensures reproducible random numbers across Python, NumPy, and PyTorch."""
        set_seed(42)
        py_r1 = random.random()
        np_r1 = np.random.rand(5)
        torch_r1 = torch.randn(5)

        set_seed(42)
        py_r2 = random.random()
        np_r2 = np.random.rand(5)
        torch_r2 = torch.randn(5)

        self.assertEqual(py_r1, py_r2, "Python random generator is not deterministic")
        self.assertTrue(np.allclose(np_r1, np_r2), "NumPy random generator is not deterministic")
        self.assertTrue(torch.allclose(torch_r1, torch_r2), "PyTorch random generator is not deterministic")

    def test_config_load(self):
        """Verify configs/config.yaml loads properly and contains all required sections and parameters."""
        cfg_path = Path("configs/config.yaml")
        self.assertTrue(cfg_path.exists(), "configs/config.yaml does not exist")

        cfg = load_config(cfg_path)
        self.assertIsInstance(cfg, dict), "Config should be parsed as a dictionary"

        # Required top-level keys
        required_keys = ["seed", "data", "augmentation", "train", "stage1", "stage2"]
        for key in required_keys:
            self.assertIn(key, cfg, f"Missing key '{key}' in config.yaml")

        # Verify data config
        self.assertIn("raw_dir", cfg["data"])
        self.assertIn("splits_dir", cfg["data"])
        self.assertGreater(cfg["data"]["image_size"], 0)

        # Verify train config
        self.assertGreater(cfg["train"]["batch_size"], 0)
        self.assertGreater(cfg["train"]["lr"], 0)
        self.assertIn("backbone", cfg["train"])


if __name__ == "__main__":
    unittest.main()
