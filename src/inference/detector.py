"""
LeafDoc Two-Stage Neural Leaf Detector Module (YOLOv8).

Integrates a specialized, lightweight YOLOv8 neural object detector pretrained on plant foliage
across 46 crop species (mAP@0.5: 0.946, 21.5 MB). Detects individual leaf bounding boxes in complex
outdoor and orchard scenes, crops out distracting branches, sky, and soil, and feeds clean leaf tissue
directly into the EfficientNet disease classifier and Grad-CAM explainability engine.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


logger = logging.getLogger("leafdoc.detector")

HF_WEIGHTS_URL = "https://huggingface.co/foduucom/plant-leaf-detection-and-classification/resolve/main/best.pt"
DEFAULT_CHECKPOINT_PATH = "checkpoints/yolov8s_leaf_detector.pt"

# Algorithm Constants for Leaf Prominence Ranking
RANK_WEIGHT_AREA: float = 0.45
RANK_WEIGHT_CENTRALITY: float = 0.35
RANK_WEIGHT_CONF: float = 0.20
MIN_BOX_AREA_RATIO: float = 0.01
MIN_BOX_DIM: int = 20
DEFAULT_PAD_PCT: float = 0.06


def download_weights(target_path: str = DEFAULT_CHECKPOINT_PATH, url: str = HF_WEIGHTS_URL) -> Path:
    """
    Ensures model weights are present locally. If missing, downloads from Hugging Face.
    """
    dest = Path(target_path)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading YOLOv8 leaf detector weights from %s to %s...", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "LeafDoc-Engine/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    logger.info("Successfully downloaded %.2f MB to %s", dest.stat().st_size / (1024 * 1024), dest)
    return dest


class LeafDetector:
    """
    Neural Leaf Object Detector wrapping YOLOv8.
    Detects individual plant leaves in an image, ranks candidate leaf blades,
    and isolates clean leaf ROI crops for subsequent disease diagnosis.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        conf_threshold: float = 0.20,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.model = None

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, torch.device):
            self.device = str(device.type)
        else:
            self.device = str(device)

        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics is not installed. LeafDetector running in fallback mode.")
            return

        ckpt_path = Path(weights_path) if weights_path else Path(DEFAULT_CHECKPOINT_PATH)
        if not ckpt_path.exists() or ckpt_path.stat().st_size < 1_000_000:
            try:
                ckpt_path = download_weights(str(ckpt_path))
            except Exception as e:
                logger.warning("Failed to download YOLO weights (%s). Running in fallback mode.", e)
                return

        try:
            self.model = YOLO(str(ckpt_path))
            self.names = getattr(self.model, "names", {})
        except Exception as e:
            logger.warning("Failed to load YOLO model from %s: %s", ckpt_path, e)
            self.model = None

    def is_available(self) -> bool:
        """Returns True if the YOLO model is loaded and ready for inference."""
        return self.model is not None

    def detect(
        self,
        image_rgb: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs neural leaf detection on an RGB image.

        Args:
            image_rgb: RGB input image as uint8 numpy array of shape (H, W, 3).
            conf_threshold: Optional confidence threshold override.

        Returns:
            List of detected candidate leaves sorted by visual prominence.
        """
        if not self.is_available() or image_rgb is None:
            return []

        h, w = image_rgb.shape[:2]
        img_area = float(h * w)
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            # Run prediction; ultralytics handles RGB numpy arrays
            results = self.model.predict(
                source=image_rgb,
                device=self.device,
                conf=thresh,
                verbose=False,
            )
        except Exception as e:
            logger.error("Leaf detection inference error: %s", e)
            return []

        if not results or len(results) == 0:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        cx_img = w / 2.0
        cy_img = h / 2.0
        max_dist = np.sqrt(cx_img**2 + cy_img**2) + 1e-6

        candidates = []
        for i, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0].tolist()
            conf = float(box.conf.cpu().numpy()[0])
            cls_id = int(box.cls.cpu().numpy()[0])
            label_name = self.names.get(cls_id, str(cls_id))

            # Clamp coordinates to image boundaries
            x1 = max(0, min(w - 1, int(round(xyxy[0]))))
            y1 = max(0, min(h - 1, int(round(xyxy[1]))))
            x2 = max(x1 + 1, min(w, int(round(xyxy[2]))))
            y2 = max(y1 + 1, min(h, int(round(xyxy[3]))))

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            # Filter microscopic false positives (< 1% of image or < 20px)
            if area < (MIN_BOX_AREA_RATIO * img_area) or bw < MIN_BOX_DIM or bh < MIN_BOX_DIM:
                continue

            # Centrality score (1.0 = center of frame, 0.0 = corner)
            bx_c = x1 + (bw / 2.0)
            by_c = y1 + (bh / 2.0)
            dist_c = np.sqrt((bx_c - cx_img) ** 2 + (by_c - cy_img) ** 2)
            centrality = max(0.1, 1.0 - (dist_c / max_dist))

            # Prominence ranking score
            area_fraction = area / img_area
            rank_score = (area_fraction * RANK_WEIGHT_AREA) + (centrality * RANK_WEIGHT_CENTRALITY) + (conf * RANK_WEIGHT_CONF)

            candidates.append({
                "index": i,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "label": label_name,
                "area": int(area),
                "width": int(bw),
                "height": int(bh),
                "rank_score": rank_score,
                "is_primary": False,
            })

        # Sort candidates by ranking score descending
        candidates.sort(key=lambda c: -c["rank_score"])

        # Re-index candidates 0 to N-1 and mark top candidate as primary
        for rank_idx, cand in enumerate(candidates):
            cand["index"] = rank_idx
            cand["is_primary"] = (rank_idx == 0)

        return candidates

    def select_primary_leaf(
        self,
        detected_leaves: List[Dict[str, Any]],
        image_shape: Tuple[int, int],
        target_species: Optional[str] = None,
        selected_box_idx: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Chooses the candidate leaf for diagnosis.
        If selected_box_idx is specified, selects that leaf.
        If target_species is provided, prioritizes matching detected classes.
        Otherwise selects candidate #0.
        """
        if not detected_leaves:
            return None

        # 1. Manual user override index
        if selected_box_idx is not None:
            for cand in detected_leaves:
                if cand["index"] == selected_box_idx:
                    for c in detected_leaves:
                        c["is_primary"] = (c["index"] == selected_box_idx)
                    return cand

        # 2. Species matching if requested
        if target_species:
            tgt_clean = target_species.lower().replace(" ", "").replace("_", "")
            for cand in detected_leaves:
                cand_clean = cand["label"].lower().replace(" ", "").replace("_", "")
                if tgt_clean in cand_clean or cand_clean in tgt_clean:
                    for c in detected_leaves:
                        c["is_primary"] = (c["index"] == cand["index"])
                    return cand

        # 3. Default to highest-ranked primary leaf
        for c in detected_leaves:
            c["is_primary"] = (c["index"] == detected_leaves[0]["index"])
        return detected_leaves[0]

    def crop_leaf_roi(
        self,
        image_rgb: np.ndarray,
        bbox: List[int],
        pad_pct: float = 0.06,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops the leaf bounding box with a margin padding to preserve edge lesions.

        Returns:
            Tuple of (cropped_rgb, (x1, y1, crop_width, crop_height)).
        """
        h, w = image_rgb.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        pad_w = int(round(bw * pad_pct))
        pad_h = int(round(bh * pad_pct))

        fx1 = max(0, x1 - pad_w)
        fy1 = max(0, y1 - pad_h)
        fx2 = min(w, x2 + pad_w)
        fy2 = min(h, y2 + pad_h)

        crop_w = fx2 - fx1
        crop_h = fy2 - fy1

        if crop_w < 16 or crop_h < 16:
            return image_rgb, (0, 0, w, h)

        cropped_rgb = image_rgb[fy1:fy2, fx1:fx2].copy()
        return cropped_rgb, (fx1, fy1, crop_w, crop_h)

    def draw_detection_boxes(
        self,
        image_rgb: np.ndarray,
        detected_leaves: List[Dict[str, Any]],
        selected_idx: Optional[int] = 0,
    ) -> np.ndarray:
        """
        Draws visual detection bounding boxes, badges, and confidence metrics on the image.

        - Primary / selected leaf: Vibrant Emerald Green (16, 185, 129)
        - Candidate secondary leaves: Sleek Amber / Cyan (245, 158, 11 / 6, 182, 212)
        """
        overlay = image_rgb.copy()
        if not detected_leaves:
            return overlay

        h, w = overlay.shape[:2]
        thickness = max(2, int(round(min(h, w) / 250)))

        for leaf in detected_leaves:
            idx = leaf["index"]
            is_selected = (idx == selected_idx) or leaf.get("is_primary", False)
            x1, y1, x2, y2 = leaf["bbox"]

            # Color scheme (RGB):
            if is_selected:
                box_color = (16, 185, 129)  # Emerald green
                text_color = (255, 255, 255)
                bg_color = (16, 185, 129)
                badge_text = f"Leaf #{idx + 1} [PRIMARY] {leaf['label'].title()} ({int(leaf['confidence'] * 100)}%)"
            else:
                box_color = (245, 158, 11)  # Warm amber
                text_color = (20, 20, 20)
                bg_color = (245, 158, 11)
                badge_text = f"Leaf #{idx + 1}: {leaf['label'].title()} ({int(leaf['confidence'] * 100)}%)"

            # Draw outer rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, thickness, lineType=cv2.LINE_AA)

            # Draw sleek corner accent brackets for selected leaf
            if is_selected:
                corner_len = max(10, int(min(x2 - x1, y2 - y1) * 0.15))
                c_thick = thickness + 2
                # Top-Left
                cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), c_thick)
                cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), c_thick)
                # Bottom-Right
                cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), c_thick)
                cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), c_thick)

            # Draw text pill badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.40, min(0.70, min(h, w) / 600.0))
            font_thick = max(1, int(round(font_scale * 2)))

            (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)
            badge_y1 = max(0, y1 - th - 8)
            badge_y2 = badge_y1 + th + 8
            badge_x2 = min(w, x1 + tw + 12)

            # Draw filled pill background
            cv2.rectangle(overlay, (x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)
            # Text inside pill
            text_pos = (x1 + 6, badge_y2 - 5)
            cv2.putText(overlay, badge_text, text_pos, font, font_scale, text_color, font_thick, lineType=cv2.LINE_AA)

        return overlay
