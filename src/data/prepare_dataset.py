"""
Scan a PlantVillage-style folder (one subfolder per class) and produce:
  - a manifest of every image with derived species / disease / binary_label / fine_label
  - a single stratified 80/10/10 split (by fine_label) written to train/val/test CSVs

Because binary_label is a deterministic function of fine_label (healthy vs not),
stratifying by fine_label automatically keeps the binary split balanced too --
no need for two separate splits, and no risk of the same image landing in
train for one task and val for the other.

Usage:
    python -m src.data.prepare_dataset --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.seed import set_seed

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def derive_labels(folder_name: str) -> tuple[str, str, str]:
    """
    PlantVillage folder names look like 'Tomato___Late_blight' or 'Apple___healthy'.
    Returns (species, disease, binary_label).
    """
    if "___" in folder_name:
        species, disease = folder_name.split("___", 1)
    else:
        # Fallback for mirrors that use a single underscore or different separator
        parts = folder_name.split("_", 1)
        species, disease = parts[0], parts[1] if len(parts) > 1 else "unknown"

    binary_label = "healthy" if disease.lower() == "healthy" else "diseased"
    return species, disease, binary_label


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    rows = []
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subfolders found in {raw_dir}. "
            "Point data.raw_dir in the config at the PlantVillage folder "
            "that directly contains one subfolder per class."
        )

    for class_dir in class_dirs:
        fine_label = class_dir.name
        species, disease, binary_label = derive_labels(fine_label)
        images = [p for p in class_dir.iterdir() if p.suffix in IMAGE_EXTENSIONS]
        for img_path in images:
            rows.append(
                {
                    "filepath": str(img_path.resolve()),
                    "species": species,
                    "disease": disease,
                    "binary_label": binary_label,
                    "fine_label": fine_label,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Found {len(class_dirs)} class folders but 0 images in {raw_dir}.")
    return df


def stratified_split(
    df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"

    # Drop any fine_label with too few images to stratify into 3 splits
    counts = df["fine_label"].value_counts()
    too_rare = counts[counts < 3].index.tolist()
    if too_rare:
        print(f"Warning: dropping {len(too_rare)} classes with <3 images (can't stratify): {too_rare}")
        df = df[~df["fine_label"].isin(too_rare)]

    train_df, temp_df = train_test_split(
        df, train_size=train_frac, stratify=df["fine_label"], random_state=seed
    )
    relative_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, train_size=relative_val, stratify=temp_df["fine_label"], random_state=seed
    )
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    raw_dir = Path(cfg["data"]["raw_dir"])
    splits_dir = Path(cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {raw_dir} ...")
    manifest = build_manifest(raw_dir)
    print(f"Found {len(manifest)} images across {manifest['fine_label'].nunique()} classes.")
    print(manifest["binary_label"].value_counts().to_string())

    train_df, val_df, test_df = stratified_split(
        manifest,
        cfg["data"]["train_frac"],
        cfg["data"]["val_frac"],
        cfg["data"]["test_frac"],
        cfg["seed"],
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_path = splits_dir / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"  {name}: {len(split_df)} images -> {out_path}")

    # Quick sanity print: per-class counts should be roughly proportional across splits
    print("\nPer-class counts (train/val/test) for first 5 classes:")
    for cls in sorted(manifest["fine_label"].unique())[:5]:
        t = (train_df["fine_label"] == cls).sum()
        v = (val_df["fine_label"] == cls).sum()
        te = (test_df["fine_label"] == cls).sum()
        print(f"  {cls}: {t}/{v}/{te}")


if __name__ == "__main__":
    main()
