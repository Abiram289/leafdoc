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
from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.augmentations import get_val_transforms
from src.models.factory import build_model
from src.utils.config import load_config


def get_target_layers(model: nn.Module, backbone_name: str = "efficientnet_b0") -> List[nn.Module]:
    """
    Identifies the appropriate target convolutional layer for Grad-CAM / LayerCAM.
    For fine-grained multi-spot lesion localization, selects feature stages with
    sufficient spatial resolution (e.g. 14x14) to resolve individual spot lesions.
    
    Args:
        model: PyTorch model instance.
        backbone_name: Architecture name (e.g. 'efficientnet_b0', 'resnet50').
        
    Returns:
        List containing the target layer module.
    """
    # 1. timm EfficientNet intermediate block with 14x14 spatial resolution (spot-level resolution)
    if hasattr(model, "blocks") and len(model.blocks) > 4:
        return [model.blocks[4]]

    # 2. timm EfficientNet conv_head
    if hasattr(model, "conv_head") and isinstance(model.conv_head, nn.Module):
        return [model.conv_head]

    # 3. timm EfficientNet bn2
    if hasattr(model, "bn2") and isinstance(model.bn2, nn.Module):
        return [model.bn2]

    # 4. ResNet layer3 / layer4
    if hasattr(model, "layer3") and len(model.layer3) > 0:
        return [model.layer3[-1]]
    if hasattr(model, "layer4") and len(model.layer4) > 0:
        return [model.layer4[-1]]

    # 5. Generic features
    if hasattr(model, "features") and len(model.features) > 0:
        return [model.features[-1]]

    # 6. Fallback: find the last Conv2d
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
    Uses LayerCAM for fine-grained multi-spot lesion localization.
    """

    def __init__(self, model: nn.Module, backbone_name: str = "efficientnet_b0", device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.backbone_name = backbone_name
        self.target_layers = get_target_layers(self.model, backbone_name)
        # LayerCAM weights element-wise positive gradients, resolving multiple individual lesion spots
        self.cam = LayerCAM(model=self.model, target_layers=self.target_layers)

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
        heatmap = np.clip(heatmap, 0.0, 1.0)
        # Multi-focal dynamic range expansion (gamma=0.5):
        # Prevents a single dominant hotspot from suppressing secondary lesion clusters
        if np.max(heatmap) > 0.0:
            heatmap = np.power(heatmap, 0.5)
        return heatmap.astype(np.float32)


def segment_leaf_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Robust foliar blade segmentation:
    Accurately isolates the complete leaf blade (healthy green, chlorotic, and necrotic areas)
    across both outdoor field clutter and laboratory backdrops (gray, white, black, purple).
    
    Args:
        image_rgb: Input RGB image (H, W, 3) in uint8 [0, 255].
        
    Returns:
        Binary mask (H, W) uint8 where 255 is leaf tissue and 0 is background.
    """
    h, w = image_rgb.shape[:2]
    img_f = image_rgb.astype(np.float32)
    R, G, B = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
    
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    exg = 2 * G - R - B

    # Analyze corner regions to detect whether the image is an outdoor field photo
    # (surrounded by vegetation/foliage) or a laboratory bench backdrop.
    cw = max(2, min(20, w // 8))
    ch = max(2, min(20, h // 8))
    corner_pixels = np.concatenate([
        img_f[:ch, :cw].reshape(-1, 3),
        img_f[:ch, -cw:].reshape(-1, 3),
        img_f[-ch:, :cw].reshape(-1, 3),
        img_f[-ch:, -cw:].reshape(-1, 3),
    ], axis=0)
    corner_exg = 2 * corner_pixels[:, 1] - corner_pixels[:, 0] - corner_pixels[:, 2]
    green_corner_pct = float(np.mean(corner_exg > 5.0) * 100.0)

    # 1. Outdoor / Cluttered Scene: Corners contain green foliage
    if green_corner_pct > 15.0:
        # Fast downsampled GrabCut to isolate the primary foreground leaf blade
        sw, sh = min(256, w), min(256, h)
        small = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        mx, my = max(2, int(sw * 0.04)), max(2, int(sh * 0.04))
        rect = (mx, my, sw - 2 * mx, sh - 2 * my)
        
        gc_mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, gc_mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except Exception:
            fg = np.ones((sh, sw), np.uint8) * 255
            
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=1)
        
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_mask = np.zeros((sh, sw), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(small_mask, [largest], -1, 255, thickness=cv2.FILLED)
        else:
            small_mask = fg
            
        full_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        full_mask = full_mask & ((V >= 25).astype(np.uint8) * 255)
        if np.count_nonzero(full_mask) > 0.05 * (h * w):
            return full_mask

    # 2. Laboratory Backdrop / Standard Bench Scene (PlantVillage standard)
    green_foliar = (exg > 5) | ((H >= 24) & (H <= 95) & (S >= 25) & (V >= 25))
    brown_foliar = (
        (R > B + 10) & (G >= B - 8) & (S >= 18) & (V >= 20) & (V <= 215)
        & ((H <= 25) | (H >= 165))
    )
    foliar_seeds = green_foliar | brown_foliar
    
    neutral_bg = (S < 16) | ((S < 25) & (V > 210)) | (V < 18)
    purple_blue_bg = (H >= 100) & (H <= 160) & (B >= G - 10)
    foliar_mask = foliar_seeds & (~neutral_bg) & (~purple_blue_bg)
    
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(foliar_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    if contours:
        min_area = 0.02 * (h * w)
        valid_c = [c for c in contours if cv2.contourArea(c) >= min_area]
        if valid_c:
            cv2.drawContours(leaf_mask, valid_c, -1, 255, thickness=cv2.FILLED)
        else:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(leaf_mask, [largest], -1, 255, thickness=cv2.FILLED)
    else:
        leaf_mask = opened
        
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, k_smooth, iterations=1)
    
    # Fill internal holes (e.g. pale necrotic centers enclosed inside the leaf)
    inv = cv2.bitwise_not(leaf_mask)
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, flood, (0, 0), 0)
    leaf_mask = leaf_mask | inv

    # Fallback sanity check: If mask is degenerate (< 10% of image), try relaxed foliar
    if np.count_nonzero(leaf_mask) < 0.10 * (h * w):
        relaxed = (exg > 0) & (~neutral_bg) & (V > 20)
        if np.count_nonzero(relaxed) > 0.10 * (h * w):
            leaf_mask = (relaxed.astype(np.uint8) * 255)
        else:
            leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
        
    return leaf_mask


def compute_severity(
    cam_heatmap: np.ndarray,
    leaf_mask: np.ndarray,
    is_healthy: bool = False,
    activation_threshold: float = 0.40,
    mild_max_pct: float = 15.0,
    moderate_max_pct: float = 40.0,
    image_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Computes disease severity and affected leaf surface percentage using dual-stream
    hybrid analysis: Grad-CAM / LayerCAM attention saliency + foliar chromatic lesion segmentation.
    
    Accurately detects both chlorotic spots (Leaf Mold, Mosaic, Spider Mites) and necrotic
    lesions (Early/Late Blight, Scab, Spots), strictly protects healthy green foliage, and
    eliminates shadow, vein, and background false positives.
    
    Args:
        cam_heatmap: 2D float32 Grad-CAM heatmap in [0.0, 1.0].
        leaf_mask: Binary mask of leaf area (255 for leaf, 0 for background).
        is_healthy: True if the diagnostic stage concluded the leaf is healthy.
        activation_threshold: Fallback threshold when image_rgb is not supplied.
        mild_max_pct: Threshold for Mild severity (< mild_max_pct).
        moderate_max_pct: Threshold for Moderate severity (mild_max_pct to moderate_max_pct).
        image_rgb: Optional RGB image array to enable chromatic lesion color segmentation.
        
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

    # Stream 2: Chromatic Foliar Lesion Analysis (when image RGB is provided)
    if image_rgb is not None:
        if image_rgb.shape[:2] != leaf_mask.shape[:2]:
            image_rgb = cv2.resize(image_rgb, (leaf_mask.shape[1], leaf_mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        img_f = image_rgb.astype(np.float32)
        R, G, B = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        b_lab = lab[:, :, 2]
        exg = 2 * G - R - B

        foliar_v = V[leaf_pixels]
        foliar_exg = exg[leaf_pixels]
        med_v = float(np.median(foliar_v)) if len(foliar_v) > 0 else 128.0
        med_exg = float(np.median(foliar_exg)) if len(foliar_exg) > 0 else 40.0

        # 1. Definitive Healthy Green Foliage:
        # High green excess index (ExG > 35 or above median), or standard foliar green with
        # strong green-over-red dominance (G > R + 18, H in [42, 98]).
        # Yellow chlorotic spots (Leaf Mold) have low G-R (G - R <= 18) and warm tone (R > B + 6).
        healthy_green = (
            leaf_pixels
            & (
                (exg > max(35.0, med_exg + 10.0))
                | ((G > R + 18) & (H >= 42) & (H <= 98) & (exg >= max(0.0, med_exg - 15.0)))
            )
        )

        # 2. Definite Physical Disease Symptoms:
        # a) Chlorosis / Leaf Mold / Pale Yellow Patches:
        # Yellowish discoloration with elevated lightness or warm chromaticity
        chlorosis_spots = (
            leaf_pixels
            & (~healthy_green)
            & (V >= 75)
            & (G - R <= 18)
            & (R > B + 6)
            & (H >= 18) & (H <= 68)
        )

        # b) Necrosis / Dark Brown / Black Spots (Early Blight, Scab, Bacterial Spot):
        # Brown/reddish dead tissue or localized dark spots
        necrosis_spots = (
            leaf_pixels
            & (~healthy_green)
            & (
                ((R > B + 10) & (R > G - 10) & ((H <= 26) | (H >= 165)) & (V >= 25) & (V <= 170))
                | ((V < med_v - 22) & (exg < med_exg - 15) & (V >= 25))
            )
        )

        # c) Powdery Mildew / Pale Mold:
        mildew_spots = (
            leaf_pixels
            & (~healthy_green)
            & (S < 35) & (V > 135) & (exg < 8) & (exg > -25)
        )

        candidate_lesions = chlorosis_spots | necrosis_spots | mildew_spots

        # Physical lesion candidates confirmed by model attention (cam >= 0.08)
        # or prominent physical lesions on the leaf blade with high chromatic contrast
        strong_color_lesions = (
            ((b_lab >= 136) & (V >= 75) & (G - R <= 16))
            | ((R > B + 14) & ((H <= 26) | (H >= 165)) & (V <= 165))
        ) & leaf_pixels & (~healthy_green)

        confirmed_lesions = (candidate_lesions & (cam_heatmap >= 0.08)) | strong_color_lesions

        # Morphological clean: remove single-pixel noise
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(confirmed_lesions.astype(np.uint8) * 255, cv2.MORPH_OPEN, k_clean)
        cleaned[~leaf_pixels] = 0
        disease_pixels = cleaned > 0
    else:
        # Stream 1 Fallback (when image_rgb not provided, e.g. synthetic test arrays)
        disease_pixels = (cam_heatmap >= activation_threshold) & leaf_pixels

    disease_count = int(np.count_nonzero(disease_pixels))
    affected_pct = (disease_count / max(total_leaf_count, 1)) * 100.0

    # Guarantee plausible minimum for confirmed diseased specimens
    if not is_healthy and disease_count > 0:
        affected_pct = max(0.1, affected_pct)

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
    leaf_mask: Optional[np.ndarray] = None,
    activation_floor: float = 0.15,
) -> np.ndarray:
    """
    Overlays a Grad-CAM heatmap onto an RGB image with adaptive transparency
    and leaf blade masking so unaffected regions and background remain natural.
    
    Args:
        image_rgb: (H, W, 3) uint8 RGB image.
        cam_heatmap: (H, W) float32 array in [0.0, 1.0].
        alpha: Maximum weight for heatmap overlay at peak activation.
        colormap: OpenCV colormap constant.
        leaf_mask: Optional binary mask of leaf area (255 for leaf, 0 for background).
                   If provided, background outside leaf is kept 100% natural.
        activation_floor: Minimum activation threshold below which overlay is transparent,
                          ensuring unaffected leaf tissue remains natural green.
        
    Returns:
        (H, W, 3) uint8 blended RGB image.
    """
    h, w = image_rgb.shape[:2]
    if cam_heatmap.shape[:2] != (h, w):
        cam_heatmap = cv2.resize(cam_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_norm = np.clip(cam_heatmap, 0.0, 1.0)
    # Ensure multi-focal dynamic range so satellite lesions are prominently displayed
    non_zero = heatmap_norm[heatmap_norm > 0.0]
    if len(non_zero) > 0 and np.mean(non_zero) < 0.35:
        heatmap_norm = np.power(heatmap_norm, 0.5)

    # Cleanly restrict heatmap to the leaf blade
    if leaf_mask is not None:
        if leaf_mask.shape[:2] != (h, w):
            leaf_mask = cv2.resize(leaf_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        leaf_binary = (leaf_mask > 0).astype(np.float32)
        heatmap_norm = heatmap_norm * leaf_binary

    # Adaptive alpha: Sub-threshold activations are transparent, keeping healthy green leaf natural.
    # Higher activations smoothly ramp up to alpha weight.
    alpha_map = np.clip((heatmap_norm - activation_floor) / max(1.0 - activation_floor, 1e-6), 0.0, 1.0) * alpha
    alpha_3ch = np.repeat(alpha_map[:, :, np.newaxis], 3, axis=2)

    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_f = image_rgb.astype(np.float32)

    overlay = alpha_3ch * heatmap_colored_rgb + (1.0 - alpha_3ch) * img_f
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
            image_rgb=img_rgb,
        )

        heatmap_bounded = np.clip(heatmap * (leaf_mask > 0).astype(np.float32), 0.0, 1.0)
        overlay = generate_gradcam_overlay(img_rgb, heatmap_bounded, leaf_mask=leaf_mask)

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
