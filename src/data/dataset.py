"""
PyTorch Dataset over the split CSVs produced by prepare_dataset.py.

One class handles both stages: pass label_col="binary_label" for Stage 1,
label_col="fine_label" for Stage 2. Label encoding (str -> int) is handled
by a shared LabelEncoder-style mapping saved alongside the split so
train/val/test and inference all agree on the same class indices.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    def __init__(self, csv_path: str, label_col: str, transforms=None, class_map_path: str | None = None):
        self.df = pd.read_csv(csv_path)
        self.label_col = label_col
        self.transforms = transforms

        classes = sorted(self.df[label_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        if class_map_path:
            Path(class_map_path).parent.mkdir(parents=True, exist_ok=True)
            if Path(class_map_path).exists():
                # reuse existing mapping (e.g. val/test reusing train's mapping)
                with open(class_map_path) as f:
                    self.class_to_idx = json.load(f)
            else:
                with open(class_map_path, "w") as f:
                    json.dump(self.class_to_idx, f, indent=2)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(row["filepath"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = self.class_to_idx[row[self.label_col]]
        return image, label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)
