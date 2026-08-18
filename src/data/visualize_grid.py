"""
Sample a grid of images from a split and save it as a PNG for a quick
by-eye sanity check: right labels, right colors, no corrupt files, and
(with --augmented) a look at what the training pipeline actually feeds the model.

Usage:
    python -m src.data.visualize_grid --config configs/config.yaml --split train --n 32
    python -m src.data.visualize_grid --config configs/config.yaml --split train --n 32 --augmented
"""
import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.augmentations import get_train_transforms
from src.utils.config import load_config
from src.utils.seed import set_seed


def denormalize(img: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--augmented", action="store_true", help="apply the training augmentation pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    csv_path = Path(cfg["data"]["splits_dir"]) / f"{args.split}.csv"
    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(args.n, len(df)), random_state=cfg["seed"])

    transforms = get_train_transforms(cfg) if args.augmented else None
    aug = cfg["augmentation"]

    cols = 8
    rows = math.ceil(len(sample) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, sample.iterrows()):
        image = cv2.imread(row["filepath"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if transforms:
            image = transforms(image=image)["image"].numpy()
            image = denormalize(image, aug["mean"], aug["std"])
        else:
            image = image / 255.0

        ax.imshow(image)
        ax.set_title(row["fine_label"], fontsize=6)
        ax.axis("off")

    for ax in axes[len(sample):]:
        ax.axis("off")

    checks_dir = Path(cfg["checks_dir"])
    checks_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_augmented" if args.augmented else ""
    out_path = checks_dir / f"{args.split}_grid{suffix}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sanity-check grid -> {out_path}")


if __name__ == "__main__":
    main()
