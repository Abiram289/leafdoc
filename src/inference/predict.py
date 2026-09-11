"""
Run the trained Stage 1, Stage 2, or full End-to-End Diagnostic Pipeline on single images,
web URLs, or entire folders of images.

Usage:
    # Full End-to-End Pipeline (Diagnosis + Grad-CAM + Severity + Treatment)
    python -m src.inference.predict --config configs/config.yaml --stage pipeline --image path/to/leaf.jpg

    # Single Stage 2 prediction
    python -m src.inference.predict --config configs/config.yaml --stage stage2 --image path/to/leaf.jpg

    # Direct Web URL with full pipeline
    python -m src.inference.predict --config configs/config.yaml --stage pipeline --image https://example.com/leaf.jpg
"""
import argparse
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.data.augmentations import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_image_from_source(source: str):
    """Loads an image from a URL or local file path into RGB numpy format."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(
            source,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                img_data = response.read()
        except Exception as e:
            raise RuntimeError(f"Failed to download image from URL ({source}): {e}")

        img_array = np.frombuffer(img_data, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image from URL: {source}")

        display_name = source.split("?")[0].split("/")[-1] or "online_image.jpg"
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb, display_name

    img_path = Path(source)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image (unsupported format or corrupt file): {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, img_path.name


def run_prediction(model, transforms, tensor_device, idx_to_class, image_rgb, display_name):
    """Runs model inference on an RGB image and prints top class predictions."""
    tensor = transforms(image=image_rgb)["image"].unsqueeze(0).to(tensor_device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    ranked = sorted(
        [(idx_to_class[i], probs[i].item()) for i in range(len(idx_to_class))],
        key=lambda x: -x[1],
    )

    print(f"\nImage: {display_name}")
    print(f"Prediction: {ranked[0][0]}  (confidence: {ranked[0][1] * 100:.1f}%)")

    if len(ranked) > 1:
        print("\nFull breakdown:")
        for label, prob in ranked[:10]:
            print(f"  {label:40s} {prob * 100:5.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Predict plant disease from image file, URL, or folder.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--stage", type=str, default="pipeline", choices=["stage1", "stage2", "pipeline"])
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a leaf image, a directory of images, or an http(s) URL",
    )
    parser.add_argument("--output-dir", type=str, default="reports/pipeline_outputs", help="Directory to save visual reports")
    args = parser.parse_args()

    # Check if target is a web URL, a directory, or a single file
    is_url = args.image.startswith(("http://", "https://"))
    target_path = Path(args.image) if not is_url else None

    if not is_url and target_path.is_dir():
        image_sources = sorted([str(p) for p in target_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        if not image_sources:
            print(f"No images found in directory: {target_path}")
            return
    else:
        image_sources = [args.image]

    if args.stage == "pipeline":
        from src.inference.pipeline import LeafDocPipeline

        pipeline = LeafDocPipeline(config_path=args.config)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Running Full LeafDoc Diagnostic Pipeline on {len(image_sources)} image(s)...")
        for src in image_sources:
            try:
                diag = pipeline.diagnose(src, generate_visualization=True)
                pipeline.print_diagnosis_report(diag)

                if diag["visualizations"]["composite_panel"] is not None:
                    out_name = f"diag_{Path(diag['image_name']).stem}.png"
                    out_path = out_dir / out_name
                    cv2.imwrite(str(out_path), cv2.cvtColor(diag["visualizations"]["composite_panel"], cv2.COLOR_RGB2BGR))
                    print(f"  Saved visual dashboard -> {out_path}\n")
            except Exception as err:
                print(f"Error processing {src}: {err}")
        return

    # Single stage inference
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

    transforms = get_val_transforms(cfg)

    print(f"Running {args.stage} inference on {len(image_sources)} image(s)...")
    for src in image_sources:
        try:
            img_rgb, name = load_image_from_source(src)
            run_prediction(model, transforms, device, idx_to_class, img_rgb, name)
            print("-" * 50)
        except Exception as err:
            print(f"Error processing {src}: {err}")


if __name__ == "__main__":
    main()

