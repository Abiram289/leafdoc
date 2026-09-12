"""
PyTorch Dataset over the split CSVs produced by prepare_dataset.py.

One class handles both stages: pass label_col="binary_label" for Stage 1,
label_col="fine_label" for Stage 2. Label encoding (str -> int) is handled
by a shared LabelEncoder-style mapping saved alongside the split so
train/val/test and inference all agree on the same class indices.
"""
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """
    PyTorch Dataset over partitioned CSV manifests for plant leaf classification.
    Supports both Stage 1 (binary) and Stage 2 (38-class fine-grained) targets.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        label_col: str,
        transforms: Optional[Callable] = None,
        class_map_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        self.transforms = transforms

        classes = sorted(self.df[label_col].unique())
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}

        if class_map_path:
            cmap_p = Path(class_map_path)
            cmap_p.parent.mkdir(parents=True, exist_ok=True)
            if cmap_p.exists():
                # Reuse existing persistent class mapping
                with open(cmap_p, "r", encoding="utf-8") as f:
                    self.class_to_idx = json.load(f)
            else:
                with open(cmap_p, "w", encoding="utf-8") as f:
                    json.dump(self.class_to_idx, f, indent=2)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Leaf image file could not be loaded or is corrupted: {filepath}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = self.class_to_idx[row[self.label_col]]
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
