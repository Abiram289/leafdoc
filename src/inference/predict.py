"""
Run the trained Stage 1 (or Stage 2, once trained) model on a single image
-- built for live demo use: point it at any image file and get a prediction
with confidence.

Usage:
    python -m src.inference.predict --config configs/config.yaml --stage stage1 --image path/to/leaf.jpg
"""
import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from src.data.augmentations import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2"])
    parser.add_argument("--image", type=str, required=True, help="path to a leaf image (jpg/png)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stage_cfg = cfg[args.stage]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg["checkpoints_dir"]) / stage_cfg["checkpoint_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path} -- train {args.stage} first.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = build_model(checkpoint["backbone"], num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image (unsupported format or corrupt file): {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transforms = get_val_transforms(cfg)
    tensor = transforms(image=image)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    # Sort all classes by probability, highest first
    ranked = sorted(
        [(idx_to_class[i], probs[i].item()) for i in range(len(idx_to_class))],
        key=lambda x: -x[1],
    )

    print(f"\nImage: {img_path.name}")
    print(f"Prediction: {ranked[0][0]}  (confidence: {ranked[0][1] * 100:.1f}%)")

    if len(ranked) > 1:
        print("\nFull breakdown:")
        for label, prob in ranked[:10]:  # top 10 max, in case this is run on a 38-class checkpoint
            print(f"  {label:40s} {prob * 100:5.1f}%")


if __name__ == "__main__":
    main()
