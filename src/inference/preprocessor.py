"""
Smart Leaf Saliency ROI Extractor and Aspect-Ratio Preserving Preprocessor.

Ensures real-world in-the-wild plant imagery (outdoor orchard photos, mobile captures)
are properly framed and undistorted before feeding into the neural classification models.
"""
from typing import Optional, Tuple
import cv2
import numpy as np


def detect_leaf_roi(
    image_rgb: np.ndarray,
    pad_pct: float = 0.08,
    min_leaf_ratio: float = 0.04,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    """
    Detects the primary leaf blade contour in an image and crops to its bounding box,
    eliminating distracting surrounding tree branches, outdoor lawn, and framing clutter.

    Args:
        image_rgb: RGB input image as uint8 numpy array (H, W, 3).
        pad_pct: Margin percentage to add around the detected leaf bounding box.
        min_leaf_ratio: Minimum fraction of total image area required to trigger cropping.

    Returns:
        Tuple of (cropped_rgb, (x1, y1, width, height), is_cropped).
    """
    h, w = image_rgb.shape[:2]
    img_area = h * w

    # Scale down for ultra-fast ROI extraction (~150ms on CPU)
    max_proc_dim = 256
    scale = min(1.0, max_proc_dim / max(h, w))
    sw = max(32, int(round(w * scale)))
    sh = max(32, int(round(h * scale)))
    small = cv2.resize(image_rgb, (sw, sh), interpolation=cv2.INTER_AREA)

    # Initial GrabCut rect: exclude outer 6% border (assumed background/framing)
    mx = max(2, int(round(sw * 0.06)))
    my = max(2, int(round(sh * 0.06)))
    rect = (mx, my, sw - 2 * mx, sh - 2 * my)

    fg_mask = None
    try:
        mask = np.zeros((sh, sw), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    except Exception:
        pass

    # Fallback if GrabCut produced degenerate mask (< 3% of small area)
    if fg_mask is None or np.count_nonzero(fg_mask) < (0.03 * sw * sh):
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
        h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        a_chan = lab[:, :, 1]
        plant_mask = (h_chan >= 24) & (h_chan <= 100) & (s_chan >= 25) & (v_chan >= 20)
        lab_green = (a_chan < 126) & (s_chan >= 25)
        fg_mask = (plant_mask | lab_green).astype(np.uint8) * 255

    # Morphological consolidation to bridge leaf lesions
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_rgb, (0, 0, w, h), False

    # Rank candidate contours: favor large, centrally located contours
    cx_s, cy_s = sw / 2.0, sh / 2.0
    max_d = np.sqrt(cx_s**2 + cy_s**2) + 1e-6
    best_score = -1.0
    best_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < (0.03 * sw * sh):
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Penalize contours touching 2 or more image boundaries (background lawn)
        touches = (bx <= 2) + (by <= 2) + (bx + bw >= sw - 2) + (by + bh >= sh - 2)
        border_penalty = 0.25 if touches >= 2 else (0.75 if touches == 1 else 1.0)

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = cx_s, cy_s

        dist = np.sqrt((cx - cx_s) ** 2 + (cy - cy_s) ** 2)
        center_w = max(0.25, 1.0 - (dist / max_d))
        score = area * center_w * border_penalty

        if score > best_score:
            best_score = score
            best_contour = c

    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    bx, by, bw, bh = cv2.boundingRect(best_contour)

    # If the detected bounding box already covers most of the image (> 82%),
    # no need to crop -- it is already a tightly framed leaf
    if (bw * bh) >= (0.82 * sw * sh) and bw >= (0.85 * sw) and bh >= (0.85 * sh):
        return image_rgb, (0, 0, w, h), False

    # Scale bbox back to original image coordinates
    orig_x1 = max(0, int(round(bx / scale)))
    orig_y1 = max(0, int(round(by / scale)))
    orig_w = min(w - orig_x1, int(round(bw / scale)))
    orig_h = min(h - orig_y1, int(round(bh / scale)))

    # Apply padding margin
    pad_w = int(round(orig_w * pad_pct))
    pad_h = int(round(orig_h * pad_pct))

    fx1 = max(0, orig_x1 - pad_w)
    fy1 = max(0, orig_y1 - pad_h)
    fx2 = min(w, orig_x1 + orig_w + pad_w)
    fy2 = min(h, orig_y1 + orig_h + pad_h)

    crop_w = fx2 - fx1
    crop_h = fy2 - fy1

    if crop_w < 32 or crop_h < 32 or (crop_w * crop_h) < (min_leaf_ratio * img_area):
        return image_rgb, (0, 0, w, h), False

    cropped_rgb = image_rgb[fy1:fy2, fx1:fx2].copy()
    return cropped_rgb, (fx1, fy1, crop_w, crop_h), True


def letterbox_image(
    image_rgb: np.ndarray,
    target_size: int = 224,
    pad_color: Tuple[int, int, int] = (128, 128, 128),
    border_mode: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    """
    Resizes an image preserving its exact aspect ratio by padding it into a square.
    Prevents horizontal/vertical stretching that distorts leaf serrations and vein angles.

    Args:
        image_rgb: Input RGB image (H, W, 3).
        target_size: Desired square output dimension (e.g. 224).
        pad_color: RGB fill color for padding borders (default is neutral gray 128, matching lab backdrop).
        border_mode: cv2 border type (cv2.BORDER_CONSTANT or cv2.BORDER_REFLECT_101).

    Returns:
        Padded square image of shape (target_size, target_size, 3).
    """
    h, w = image_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Choose interpolation: INTER_AREA when shrinking, INTER_LINEAR when enlarging
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=interp)

    # Compute symmetric padding
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    if border_mode == cv2.BORDER_CONSTANT:
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    else:
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=border_mode,
        )

    return padded


def preprocess_leaf(
    image_rgb: np.ndarray,
    auto_crop: bool = True,
    target_size: int = 224,
    pad_color: Tuple[int, int, int] = (128, 128, 128),
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], bool]:
    """
    Unified end-to-end preprocessor:
    1. Detects primary leaf blade and crops out surrounding branches/grass (if auto_crop=True).
    2. Letterboxes the leaf into a square preserving its natural aspect ratio.

    Returns:
        Tuple of (letterboxed_rgb, cropped_leaf_rgb, (x1, y1, w, h), is_cropped).
    """
    if auto_crop:
        cropped_leaf, bbox, is_cropped = detect_leaf_roi(image_rgb)
    else:
        cropped_leaf = image_rgb
        bbox = (0, 0, image_rgb.shape[1], image_rgb.shape[0])
        is_cropped = False

    letterboxed = letterbox_image(cropped_leaf, target_size=target_size, pad_color=pad_color)
    return letterboxed, cropped_leaf, bbox, is_cropped
