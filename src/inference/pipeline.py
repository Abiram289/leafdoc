"""
LeafDoc Unified End-to-End Plant Disease Diagnosis Pipeline.

Orchestrates all 4 stages:
  1. Stage 1: Binary Classifier (Healthy vs Diseased)
  2. Stage 2: Fine-Grained Classifier (38 PlantVillage Species & Diseases)
  3. Stage 3: Explainable AI & Severity (Grad-CAM Heatmap, Leaf Segmentation, % Affected Surface Area)
  4. Stage 4: Agronomic Treatment Recommender (Cultural, Chemical, Biological Remedies)
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.augmentations import get_val_transforms
from src.inference.gradcam import (
    GradCAMExplainer,
    compute_severity,
    generate_annotated_panel,
    generate_gradcam_overlay,
    segment_leaf_mask,
)
from src.inference.predict import load_image_from_source
from src.inference.preprocessor import preprocess_leaf
from src.inference.treatments import TreatmentRecommender
from src.models.factory import build_model
from src.utils.config import load_config


class LeafDocPipeline:
    """
    Unified end-to-end multi-stage inference pipeline for plant leaf diagnosis.
    """

    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        stage1_ckpt_path: Optional[str] = None,
        stage2_ckpt_path: Optional[str] = None,
        treatments_path: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.cfg = load_config(config_path)
        
        # Setup compute device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        checkpoints_dir = Path(self.cfg["checkpoints_dir"])
        
        # 1. Load Stage 1 Binary Classifier
        s1_path = Path(stage1_ckpt_path) if stage1_ckpt_path else checkpoints_dir / self.cfg["stage1"]["checkpoint_name"]
        if not s1_path.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found at: {s1_path}")
        
        s1_ckpt = torch.load(s1_path, map_location=self.device, weights_only=False)
        self.s1_class_to_idx = s1_ckpt["class_to_idx"]
        self.s1_idx_to_class = {v: k for k, v in self.s1_class_to_idx.items()}
        self.s1_model = build_model(s1_ckpt["backbone"], num_classes=len(self.s1_class_to_idx)).to(self.device)
        self.s1_model.load_state_dict(s1_ckpt["model_state_dict"])
        self.s1_model.eval()

        # 2. Load Stage 2 Fine Classifier
        s2_path = Path(stage2_ckpt_path) if stage2_ckpt_path else checkpoints_dir / self.cfg["stage2"]["checkpoint_name"]
        if not s2_path.exists():
            raise FileNotFoundError(f"Stage 2 checkpoint not found at: {s2_path}")

        s2_ckpt = torch.load(s2_path, map_location=self.device, weights_only=False)
        self.s2_class_to_idx = s2_ckpt["class_to_idx"]
        self.s2_idx_to_class = {v: k for k, v in self.s2_class_to_idx.items()}
        self.s2_backbone = s2_ckpt["backbone"]
        self.s2_model = build_model(self.s2_backbone, num_classes=len(self.s2_class_to_idx)).to(self.device)
        self.s2_model.load_state_dict(s2_ckpt["model_state_dict"])
        self.s2_model.eval()

        # 3. Setup Explainability Engine (Grad-CAM on Stage 2)
        self.explainer = GradCAMExplainer(self.s2_model, backbone_name=self.s2_backbone, device=self.device)

        # 4. Setup Treatment Recommender
        self.recommender = TreatmentRecommender(treatments_path=treatments_path)

        # 5. Transforms & Thresholds
        self.transforms = get_val_transforms(self.cfg)
        self.mild_max_pct = float(self.cfg.get("severity", {}).get("mild_max_pct", 15.0))
        self.moderate_max_pct = float(self.cfg.get("severity", {}).get("moderate_max_pct", 40.0))

    def _prepare_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, str]:
        """Converts diverse image input formats into an RGB NumPy array."""
        if isinstance(image_input, (str, Path)):
            img_rgb, name = load_image_from_source(str(image_input))
            return img_rgb, name
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
            return img_rgb, "pil_image.jpg"
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim != 3 or image_input.shape[2] != 3:
                raise ValueError(f"Expected 3-channel RGB image, got shape {image_input.shape}")
            return image_input, "array_image.jpg"
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def get_supported_species(self) -> List[str]:
        """Returns a sorted list of unique human-friendly plant species supported by the model."""
        species_set = set()
        for label in self.s2_class_to_idx.keys():
            if "___" in label:
                sp = label.split("___", 1)[0].replace("_", " ")
            else:
                sp = label.replace("_", " ")
            species_set.add(sp)
        return sorted(list(species_set))

    def _match_species_indices(self, target_species: Optional[str]) -> Tuple[List[int], Optional[str]]:
        """
        Finds Stage 2 class indices matching the target plant species.
        Supports flexible matching (e.g., 'Potato', 'tomato', 'Corn', 'Pepper', 'Cherry').
        """
        if not target_species:
            return [], None

        target_clean = (
            target_species.lower()
            .replace("_", "")
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )
        if not target_clean or target_clean in ("all", "auto", "autodetect", "none"):
            return [], None

        matching_indices = []
        matched_species_names = set()

        for idx, class_name in self.s2_idx_to_class.items():
            if "___" in class_name:
                species_raw = class_name.split("___", 1)[0]
            else:
                species_raw = class_name

            spec_clean = (
                species_raw.lower()
                .replace("_", "")
                .replace(" ", "")
                .replace(",", "")
                .replace("(", "")
                .replace(")", "")
                .strip()
            )

            if target_clean == spec_clean or target_clean in spec_clean or spec_clean in target_clean:
                matching_indices.append(idx)
                matched_species_names.add(species_raw.replace("_", " "))

        canonical_name = next(iter(matched_species_names)) if matched_species_names else target_species
        return matching_indices, canonical_name

    def diagnose(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        generate_visualization: bool = True,
        top_k: int = 5,
        auto_crop: bool = True,
        target_species: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full 4-stage diagnosis on an input leaf image.
        
        Args:
            image_input: File path, URL, PIL image, or RGB numpy array.
            generate_visualization: If True, computes Grad-CAM overlay and 4-panel dashboard.
            top_k: Number of top fine-grained predictions to return.
            auto_crop: If True, automatically detects primary leaf ROI and crops out outdoor clutter.
            target_species: Optional plant species constraint (e.g. 'Potato', 'Tomato', 'Apple').
                            When provided, locks Stage 2 inference to diseases of that plant species.
            
        Returns:
            Structured dictionary with outputs from all 4 stages.
        """
        img_rgb, display_name = self._prepare_image(image_input)

        # Smart Leaf ROI detection and aspect-ratio preserving letterboxing
        target_dim = self.cfg["data"].get("image_size", 224)
        letterboxed_rgb, cropped_leaf_rgb, roi_bbox, is_cropped = preprocess_leaf(
            image_rgb=img_rgb,
            auto_crop=auto_crop,
            target_size=target_dim,
        )

        # Build normalized tensor from aspect-preserving letterboxed image
        tensor = self.transforms(image=letterboxed_rgb)["image"].unsqueeze(0).to(self.device)

        # -------------------------------------------------------------
        # Stage 1: Binary Classifier (Healthy vs Diseased)
        # -------------------------------------------------------------
        with torch.no_grad():
            s1_logits = self.s1_model(tensor)
            s1_probs = F.softmax(s1_logits, dim=1)[0]
            s1_top_idx = int(torch.argmax(s1_probs).item())
            s1_pred_label = self.s1_idx_to_class[s1_top_idx]
            s1_confidence = float(s1_probs[s1_top_idx].item())

        is_diseased = (s1_pred_label.lower() == "diseased")
        s1_prob_dict = {self.s1_idx_to_class[i]: float(s1_probs[i].item()) for i in range(len(self.s1_idx_to_class))}

        # -------------------------------------------------------------
        # Stage 2: Fine-Grained Classifier (38 classes or Target Species Filter)
        # -------------------------------------------------------------
        with torch.no_grad():
            s2_logits = self.s2_model(tensor)
            s2_probs = F.softmax(s2_logits, dim=1)[0]

        species_indices, matched_species_name = self._match_species_indices(target_species)

        if species_indices:
            # Crop/Species-conditioned inference: isolate subset and re-normalize probabilities
            subset_probs = torch.tensor([s2_probs[i].item() for i in species_indices], device=self.device)
            prob_sum = float(subset_probs.sum().item())
            if prob_sum > 1e-7:
                renorm_probs = subset_probs / prob_sum
            else:
                renorm_probs = torch.ones_like(subset_probs) / len(species_indices)

            best_local_idx = int(torch.argmax(renorm_probs).item())
            s2_top_idx = species_indices[best_local_idx]
            s2_pred_label = self.s2_idx_to_class[s2_top_idx]
            s2_confidence = float(renorm_probs[best_local_idx].item())

            ranked_s2 = sorted(
                [
                    (self.s2_idx_to_class[species_indices[j]], float(renorm_probs[j].item()))
                    for j in range(len(species_indices))
                ],
                key=lambda x: -x[1],
            )
            top_k_list = [{"label": label, "confidence": round(prob, 4)} for label, prob in ranked_s2[:top_k]]
            applied_species = matched_species_name
        else:
            # Standard unconstrained 38-class inference
            s2_top_idx = int(torch.argmax(s2_probs).item())
            s2_pred_label = self.s2_idx_to_class[s2_top_idx]
            s2_confidence = float(s2_probs[s2_top_idx].item())

            ranked_s2 = sorted(
                [(self.s2_idx_to_class[i], float(s2_probs[i].item())) for i in range(len(self.s2_idx_to_class))],
                key=lambda x: -x[1],
            )
            top_k_list = [{"label": label, "confidence": round(prob, 4)} for label, prob in ranked_s2[:top_k]]
            applied_species = None

        # Parse species and condition
        if "___" in s2_pred_label:
            species_raw, disease_raw = s2_pred_label.split("___", 1)
        else:
            species_raw, disease_raw = "Unknown", s2_pred_label

        species = species_raw.replace("_", " ")
        disease_clean = disease_raw.replace("_", " ")
        is_fine_healthy = "healthy" in disease_raw.lower()

        # Consensus check: if Stage 1 strongly says healthy and top fine class is healthy
        is_overall_healthy = is_fine_healthy or (not is_diseased and s1_confidence > 0.90)

        # -------------------------------------------------------------
        # Stage 3: Grad-CAM Explainability & Severity Assessment
        # -------------------------------------------------------------
        cam_heatmap = self.explainer.generate_heatmap(tensor, target_class_idx=s2_top_idx)
        subject_rgb = cropped_leaf_rgb if is_cropped else img_rgb
        leaf_mask = segment_leaf_mask(subject_rgb)

        severity_info = compute_severity(
            cam_heatmap=cam_heatmap,
            leaf_mask=leaf_mask,
            is_healthy=is_overall_healthy,
            mild_max_pct=self.mild_max_pct,
            moderate_max_pct=self.moderate_max_pct,
        )

        # -------------------------------------------------------------
        # Stage 4: Agronomic Treatment Recommendation
        # -------------------------------------------------------------
        treatment_info = self.recommender.get_recommendation(
            fine_label=s2_pred_label,
            severity=severity_info["severity"],
        )
        treatment_summary_text = self.recommender.format_text_summary(
            fine_label=s2_pred_label,
            severity=severity_info["severity"],
        )

        # -------------------------------------------------------------
        # Visualizations (Grad-CAM Overlay & 4-Panel Dashboard)
        # -------------------------------------------------------------
        cam_overlay = None
        composite_panel = None
        if generate_visualization:
            cam_overlay = generate_gradcam_overlay(subject_rgb, cam_heatmap)
            composite_panel = generate_annotated_panel(
                image_rgb=subject_rgb,
                cam_overlay=cam_overlay,
                disease_mask=severity_info["disease_mask"],
                species=species,
                condition_name=treatment_info.get("condition_name", disease_clean),
                confidence=s2_confidence,
                severity=severity_info["severity"],
                affected_pct=severity_info["affected_area_pct"],
            )

        # Build output structure
        return {
            "image_name": display_name,
            "is_cropped": is_cropped,
            "roi_bbox": list(roi_bbox),
            "target_species": applied_species,
            "is_healthy": is_overall_healthy,
            "overall_status": "Healthy" if is_overall_healthy else "Diseased",
            "stage1_binary": {
                "label": s1_pred_label,
                "confidence": round(s1_confidence, 4),
                "is_diseased": is_diseased,
                "probabilities": s1_prob_dict,
            },
            "stage2_fine": {
                "fine_label": s2_pred_label,
                "species": species,
                "disease": disease_clean,
                "condition_name": treatment_info.get("condition_name", disease_clean),
                "confidence": round(s2_confidence, 4),
                "top_predictions": top_k_list,
                "target_species_applied": applied_species,
            },
            "stage3_severity": {
                "severity": severity_info["severity"],
                "affected_area_pct": severity_info["affected_area_pct"],
                "total_leaf_pixels": severity_info["total_leaf_pixels"],
                "affected_pixels": severity_info["affected_pixels"],
            },
            "stage4_treatment": treatment_info,
            "treatment_summary_text": treatment_summary_text,
            "visualizations": {
                "original_rgb": img_rgb if generate_visualization else None,
                "cropped_leaf": cropped_leaf_rgb if (generate_visualization and is_cropped) else None,
                "cam_heatmap": cam_heatmap if generate_visualization else None,
                "cam_overlay": cam_overlay,
                "disease_mask": severity_info["disease_mask"] if generate_visualization else None,
                "composite_panel": composite_panel,
            },
        }

    def print_diagnosis_report(self, diagnosis: Dict[str, Any]):
        """Prints a well-formatted CLI summary of the full diagnosis."""
        s2 = diagnosis["stage2_fine"]
        s3 = diagnosis["stage3_severity"]
        t = diagnosis["stage4_treatment"]

        print("=" * 72)
        print("                  LEAFDOC COMPLETE DIAGNOSTIC REPORT")
        print("=" * 72)
        print(f"Image Source     : {diagnosis['image_name']}")
        print(f"Health Status    : {diagnosis['overall_status']}")
        print(f"Plant Species    : {s2['species']}")
        print(f"Diagnosis        : {s2['condition_name']} ({s2['fine_label']})")
        print(f"Confidence       : {s2['confidence'] * 100:.2f}%")
        print(f"Severity Rating  : {s3['severity']}")
        print(f"Affected Area    : {s3['affected_area_pct']:.2f}%")
        print("-" * 72)
        print(f"Action Required  : {t.get('immediate_action', 'None')}")
        print("-" * 72)
        print(diagnosis["treatment_summary_text"])
        print("=" * 72)
