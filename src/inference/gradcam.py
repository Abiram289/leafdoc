"""
Stage 3: Grad-CAM Explainability, Leaf Area Segmentation, and Severity Assessment.

Extracts class activation maps (Grad-CAM) from the fine-grained model, segments the leaf
surface area, calculates the percentage of affected leaf tissue, and classifies disease
severity (Healthy, Mild, Moderate, Severe).
"""
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.augmentations import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config


def get_target_layers(model: nn.Module, backbone_name: str = "efficientnet_b0") -> List[nn.Module]:
    """
    Identifies the appropriate target convolutional layer for Grad-CAM.
    
    Args:
        model: PyTorch model instance.
        backbone_name: Architecture name (e.g. 'efficientnet_b0', 'resnet50').
        
    Returns:
        List containing the target layer module.
    """
    # 1. Check for timm EfficientNet conv_head
    if hasattr(model, "conv_head") and isinstance(model.conv_head, nn.Module):
        return [model.conv_head]

    # 2. Check for ResNet layer4
    if hasattr(model, "layer4") and len(model.layer4) > 0:
        return [model.layer4[-1]]

    # 3. Check for features[-1] (common in many timm architectures)
    if hasattr(model, "features") and len(model.features) > 0:
        last_block = model.features[-1]
        return [last_block]

    # 4. Fallback: find the last nn.Conv2d module in the network
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is not None:
        return [last_conv]

    raise ValueError(f"Could not automatically determine target layer for backbone '{backbone_name}'.")


class GradCAMExplainer:
    """
    Wrapper around pytorch-grad-cam for computing normalized saliency heatmaps.
    """

    def __init__(self, model: nn.Module, backbone_name: str = "efficientnet_b0", device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.backbone_name = backbone_name
        self.target_layers = get_target_layers(self.model, backbone_name)
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_heatmap(
        self,
        tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a 2D float32 Grad-CAM heatmap in [0.0, 1.0].
        
        Args:
            tensor: Preprocessed image tensor with shape (1, 3, H, W) on self.device.
            target_class_idx: Target class index to explain. If None, uses top predicted class.
            
        Returns:
            2D numpy array of shape (H, W) with values in [0.0, 1.0].
        """
        targets = [ClassifierOutputTarget(target_class_idx)] if target_class_idx is not None else None
        grayscale_cam = self.cam(input_tensor=tensor, targets=targets)
        # grayscale_cam has shape (1, H, W)
        heatmap = grayscale_cam[0, :]
        return np.clip(heatmap, 0.0, 1.0)


def segment_leaf_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Segments the leaf region from neutral/white/gray/dark backgrounds.
    
    Args:
        image_rgb: Input RGB image (H, W, 3) in uint8 [0, 255].
        
    Returns:
        Binary mask (H, W) uint8 where 255 is leaf tissue and 0 is background.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    
    # 1. Saturation channel in HSV (leaves have color saturation, neutral gray/white backgrounds have very low S)
    s_channel = hsv[:, :, 1]
    
    # 2. Chromatic distance in LAB space: |a - 128| + |b - 128|
    a_diff = np.abs(lab[:, :, 1].astype(np.float32) - 128.0)
    b_diff = np.abs(lab[:, :, 2].astype(np.float32) - 128.0)
    chroma_dist = cv2.normalize(a_diff + b_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Combined color saliency
    combined_saliency = cv2.addWeighted(s_channel, 0.5, chroma_dist, 0.5, 0)
    
    # Otsu thresholding on combined saliency
    _, binary_mask = cv2.threshold(combined_saliency, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Exclude extreme specular highlights / pure white edges and pitch-black border pixels
    v_channel = hsv[:, :, 2]
    valid_lum = (v_channel > 20) & (v_channel < 250)
    binary_mask = binary_mask & (valid_lum.astype(np.uint8) * 255)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Fill small holes inside connected components
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_cleaned = np.zeros_like(binary_mask)
    if contours:
        # Keep contours that are at least 1% of the image area
        min_area = 0.01 * (image_rgb.shape[0] * image_rgb.shape[1])
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        if valid_contours:
            cv2.drawContours(mask_cleaned, valid_contours, -1, 255, thickness=cv2.FILLED)
        else:
            # If no large contour, use the largest single contour
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask_cleaned, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        mask_cleaned = binary_mask

    # Fallback sanity check: If mask is degenerate (< 5% of image area), assume the whole image is leaf
    leaf_ratio = np.count_nonzero(mask_cleaned) / (image_rgb.shape[0] * image_rgb.shape[1])
    if leaf_ratio < 0.05 or leaf_ratio > 0.98:
        mask_cleaned = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8) * 255

    return mask_cleaned


def compute_severity(
    cam_heatmap: np.ndarray,
    leaf_mask: np.ndarray,
    is_healthy: bool = False,
    activation_threshold: float = 0.50,
    mild_max_pct: float = 15.0,
    moderate_max_pct: float = 40.0,
) -> Dict[str, Any]:
    """
    Computes disease severity and affected leaf surface percentage.
    
    Args:
        cam_heatmap: 2D float32 Grad-CAM heatmap in [0.0, 1.0].
        leaf_mask: Binary mask of leaf area (255 for leaf, 0 for background).
        is_healthy: True if the diagnostic stage concluded the leaf is healthy.
        activation_threshold: Threshold above which heatmap activation indicates disease lesion.
        mild_max_pct: Threshold for Mild severity (< mild_max_pct).
        moderate_max_pct: Threshold for Moderate severity (mild_max_pct to moderate_max_pct).
        
    Returns:
        Dict with affected area %, severity label, counts, and disease mask.
    """
    # Resize heatmap if shapes differ
    if cam_heatmap.shape[:2] != leaf_mask.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (leaf_mask.shape[1], leaf_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

    if is_healthy:
        return {
            "affected_area_pct": 0.0,
            "severity": "Healthy",
            "total_leaf_pixels": int(np.count_nonzero(leaf_mask)),
            "affected_pixels": 0,
            "disease_mask": np.zeros_like(leaf_mask, dtype=np.uint8),
            "activation_threshold": activation_threshold,
        }

    leaf_pixels = leaf_mask > 0
    total_leaf_count = int(np.count_nonzero(leaf_pixels))
    if total_leaf_count == 0:
        total_leaf_count = leaf_mask.size
        leaf_pixels = np.ones_like(leaf_mask, dtype=bool)

    # Disease lesion region: high activation within segmented leaf
    disease_pixels = (cam_heatmap >= activation_threshold) & leaf_pixels
    disease_count = int(np.count_nonzero(disease_pixels))

    affected_pct = (disease_count / max(total_leaf_count, 1)) * 100.0

    if affected_pct < mild_max_pct:
        severity = "Mild"
    elif affected_pct <= moderate_max_pct:
        severity = "Moderate"
    else:
        severity = "Severe"

    disease_mask = (disease_pixels.astype(np.uint8) * 255)

    return {
        "affected_area_pct": round(float(affected_pct), 2),
        "severity": severity,
        "total_leaf_pixels": total_leaf_count,
        "affected_pixels": disease_count,
        "disease_mask": disease_mask,
        "activation_threshold": activation_threshold,
    }


def generate_gradcam_overlay(
    image_rgb: np.ndarray,
    cam_heatmap: np.ndarray,
    alpha: float = 0.55,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto an RGB image.
    
    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        cam_heatmap: (H, W) float32 array in [0.0, 1.0].
        alpha: Weight for heatmap overlay.
        colormap: OpenCV colormap constant.
        
    Returns:
        (H, W, 3) uint8 blended RGB image.
    """
    if cam_heatmap.shape[:2] != image_rgb.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.uint8(255 * np.clip(cam_heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap_colored_rgb + (1.0 - alpha) * image_rgb.astype(np.float32))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_annotated_panel(
    image_rgb: np.ndarray,
    cam_overlay: np.ndarray,
    disease_mask: np.ndarray,
    species: str,
    condition_name: str,
    confidence: float,
    severity: str,
    affected_pct: float,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Creates a professional 4-panel diagnostic dashboard image.
    
    Panels:
        1. Original Leaf Image
        2. Grad-CAM Saliency Overlay
        3. Segmented Disease Lesion Mask
        4. Diagnostic Information Card
        
    Returns:
        (H, W, 3) uint8 composite RGB image.
    """
    h, w = target_size
    img_resized = cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)
    cam_resized = cv2.resize(cam_overlay, (w, h), interpolation=cv2.INTER_AREA)

    # Highlight disease mask in crimson red overlay on grayscale leaf
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    mask_resized = cv2.resize(disease_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    lesion_vis = gray_3ch.copy()
    lesion_pixels = mask_resized > 0
    if np.any(lesion_pixels):
        # Apply red highlight for lesions
        lesion_vis[lesion_pixels] = (lesion_vis[lesion_pixels] * 0.3 + np.array([230, 45, 45]) * 0.7).astype(np.uint8)

    # Panel 4: Diagnostic Card
    card = np.full((h, w * 3 // 2, 3), 24, dtype=np.uint8)  # Sleek dark slate background (24, 24, 24)

    # Severity color badge
    if severity.lower() == "healthy":
        badge_color = (46, 204, 113)   # Emerald Green
    elif severity.lower() == "mild":
        badge_color = (52, 152, 219)   # Sky Blue
    elif severity.lower() == "moderate":
        badge_color = (243, 156, 18)   # Amber Orange
    else:
        badge_color = (231, 76, 60)    # Crimson Red

    # Header
    cv2.putText(card, "LEAFDOC DIAGNOSIS", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(card, (16, 42), (card.shape[1] - 16, 42), (60, 60, 65), 1)

    # Species & Condition
    cv2.putText(card, "Plant Species:", (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(card, f"{species}", (16, 88), cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(card, "Diagnosis:", (16, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    # Truncate condition text if very long
    cond_disp = condition_name if len(condition_name) <= 24 else condition_name[:22] + "..."
    cv2.putText(card, f"{cond_disp}", (16, 134), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Confidence Score
    cv2.putText(card, f"Confidence: {confidence * 100:.1f}%", (16, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Severity Badge & Affected Area
    cv2.putText(card, "Severity Level:", (16, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.rectangle(card, (16, 196), (135, 222), badge_color, -1)
    cv2.putText(card, severity.upper(), (24, 214), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(card, f"Affected: {affected_pct:.1f}%", (145, 214), cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)

    # Affected area progress bar
    bar_x, bar_y, bar_w, bar_h = 16, 232, card.shape[1] - 32, 10
    cv2.rectangle(card, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 55), -1)
    fill_w = int(bar_w * min(affected_pct / 100.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(card, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), badge_color, -1)

    # Add titles to individual image panels
    for img, title in [(img_resized, "1. Input Leaf"), (cam_resized, "2. Grad-CAM"), (lesion_vis, "3. Lesion Mask")]:
        cv2.rectangle(img, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(img, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Horizontal stack: Image | Grad-CAM | Lesion Mask | Info Card
    composite = np.hstack([img_resized, cam_resized, lesion_vis, card])
    return composite


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmap and severity assessment for leaf images.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--image", type=str, required=True, help="Path to leaf image or directory of images")
    parser.add_argument("--output-dir", type=str, default="reports/gradcam_outputs", help="Directory to save visual outputs")
    parser.add_argument("--save-composite", action="store_true", default=True, help="Save 4-panel diagnostic dashboard")
    args = parser.parse_args()

    from src.inference.predict import IMAGE_EXTENSIONS, load_image_from_source

    cfg = load_config(args.config)
    stage2_cfg = cfg["stage2"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(cfg["checkpoints_dir"]) / stage2_cfg["checkpoint_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train stage 2 first.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    backbone = checkpoint["backbone"]

    model = build_model(backbone, num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transforms = get_val_transforms(cfg)
    explainer = GradCAMExplainer(model, backbone_name=backbone, device=device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = Path(args.image)
    if target_path.is_dir():
        image_files = sorted([p for p in target_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    else:
        image_files = [target_path]

    print(f"Processing {len(image_files)} image(s) with Grad-CAM & Severity Analysis...")
    for img_p in image_files:
        img_rgb, name = load_image_from_source(str(img_p))
        tensor = transforms(image=image_rgb)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            top_prob = float(probs[top_idx].item())

        predicted_label = idx_to_class[top_idx]
        is_healthy = "healthy" in predicted_label.lower()

        # Extract Grad-CAM heatmap
        heatmap = explainer.generate_heatmap(tensor, target_class_idx=top_idx)
        leaf_mask = segment_leaf_mask(img_rgb)
        
        severity_result = compute_severity(
            cam_heatmap=heatmap,
            leaf_mask=leaf_mask,
            is_healthy=is_healthy,
            mild_max_pct=cfg.get("severity", {}).get("mild_max_pct", 15.0),
            moderate_max_pct=cfg.get("severity", {}).get("moderate_max_pct", 40.0),
        )

        overlay = generate_gradcam_overlay(img_rgb, heatmap)

        species = predicted_label.split("___")[0].replace("_", " ") if "___" in predicted_label else "Plant"
        disease = predicted_label.split("___")[1].replace("_", " ") if "___" in predicted_label else predicted_label

        print(f"\nImage: {name}")
        print(f"  Prediction : {predicted_label} ({top_prob * 100:.1f}%)")
        print(f"  Severity   : {severity_result['severity']}")
        print(f"  Affected % : {severity_result['affected_area_pct']:.2f}%")

        if args.save_composite:
            panel = generate_annotated_panel(
                image_rgb=img_rgb,
                cam_overlay=overlay,
                disease_mask=severity_result["disease_mask"],
                species=species,
                condition_name=disease,
                confidence=top_prob,
                severity=severity_result["severity"],
                affected_pct=severity_result["affected_area_pct"],
            )
            out_file = out_dir / f"gradcam_{Path(name).stem}.png"
            cv2.imwrite(str(out_file), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"  Saved viz  : {out_file}")


if __name__ == "__main__":
    main()
