"""
FastAPI REST API routes for LeafDoc.
"""
import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status

from app.schemas import (
    ClassesResponse,
    DetectedLeafItem,
    DiagnosisResponse,
    HealthResponse,
    SampleImageItem,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    Stage4Result,
    VisualizationsResult,
)
from src.inference.pipeline import LeafDocPipeline

router = APIRouter(prefix="/api/v1", tags=["LeafDoc API"])


def encode_array_to_base64(array: Optional[np.ndarray], quality: int = 92) -> Optional[str]:
    """Encodes a 2D or 3D numpy array to a base64 Data URI."""
    if array is None:
        return None
    try:
        # Ensure uint8
        if array.dtype != np.uint8:
            if array.max() <= 1.0:
                u8_arr = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                u8_arr = np.clip(array, 0, 255).astype(np.uint8)
        else:
            u8_arr = array

        if u8_arr.ndim == 2:
            # Grayscale / mask
            success, buf = cv2.imencode(".png", u8_arr)
            mime = "image/png"
        elif u8_arr.ndim == 3 and u8_arr.shape[2] == 3:
            # RGB array to BGR for cv2
            bgr = cv2.cvtColor(u8_arr, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            mime = "image/jpeg"
        else:
            return None

        if not success:
            return None
        encoded = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception as e:
        print(f"Base64 encoding error: {e}")
        return None


def get_pipeline(request: Request) -> LeafDocPipeline:
    """Retrieves the preloaded singleton LeafDocPipeline from app state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        # Fallback initialization if lifespan didn't populate (e.g. ad-hoc router tests)
        pipeline = LeafDocPipeline()
        request.app.state.pipeline = pipeline
    return pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint returning system status, device acceleration, and model checkpoint health.
    """
    pipeline = get_pipeline(request)
    device_type = pipeline.device.type
    import torch
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Host CPU"

    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=device_type,
        device_name=device_name,
        stage1_checkpoint="Loaded (Binary EfficientNet-B0)",
        stage2_checkpoint="Loaded (38-Class EfficientNet-B0)",
        total_classes=len(pipeline.s2_idx_to_class),
        total_treatments=len(pipeline.recommender.treatments),
    )


@router.get("/classes", response_model=ClassesResponse)
async def list_classes(request: Request):
    """
    Returns all 38 fine-grained plant species and disease classes supported by LeafDoc.
    """
    pipeline = get_pipeline(request)
    classes_list = []
    species_set = set()

    for idx, fine_label in sorted(pipeline.s2_idx_to_class.items()):
        if "___" in fine_label:
            species_raw, cond_raw = fine_label.split("___", 1)
        else:
            species_raw, cond_raw = "Unknown", fine_label

        species = species_raw.replace("_", " ")
        cond = cond_raw.replace("_", " ")
        is_healthy = "healthy" in cond_raw.lower()
        species_set.add(species)

        classes_list.append({
            "fine_label": fine_label,
            "species": species,
            "condition": cond,
            "is_healthy": is_healthy,
        })

    return ClassesResponse(
        total_classes=len(classes_list),
        classes=classes_list,
        species_list=sorted(list(species_set)),
    )


@router.get("/treatments")
async def get_treatments(request: Request, species: Optional[str] = Query(None)):
    """
    Returns the treatment encyclopedia, optionally filtered by plant species.
    """
    pipeline = get_pipeline(request)
    all_treatments = pipeline.recommender.treatments

    if species:
        spec_clean = species.strip().lower()
        filtered = {k: v for k, v in all_treatments.items() if spec_clean in v.get("species", "").lower()}
        return {"total": len(filtered), "species_filter": species, "treatments": filtered}

    return {"total": len(all_treatments), "treatments": all_treatments}


@router.get("/treatments/{fine_label}")
async def get_treatment_by_label(fine_label: str, request: Request):
    """
    Returns full agronomic treatment record for a specific disease class.
    """
    pipeline = get_pipeline(request)
    treatment = pipeline.recommender.get_recommendation(fine_label=fine_label)
    if not treatment:
        raise HTTPException(status_code=404, detail=f"No treatment entry found for '{fine_label}'")
    return treatment


@router.get("/samples", response_model=List[SampleImageItem])
async def list_sample_images():
    """
    Returns a curated list of demonstration leaf samples for instant 1-click UI diagnosis.
    """
    samples_dir = Path("app/static/samples")
    samples = []
    
    predefined = [
        {
            "id": "apple-scab",
            "title": "Apple Scab (Venturia inaequalis)",
            "species": "Apple",
            "expected_condition": "Apple Scab",
            "filename": "apple_scab.jpg",
        },
        {
            "id": "peach-healthy",
            "title": "Peach Leaf (Vigorous & Healthy)",
            "species": "Peach",
            "expected_condition": "Healthy",
            "filename": "peach_healthy.jpg",
        },
        {
            "id": "tomato-mold",
            "title": "Tomato Leaf Mold (Passalora fulva)",
            "species": "Tomato",
            "expected_condition": "Leaf Mold",
            "filename": "tomato_leaf_mold.jpg",
        },
        {
            "id": "tomato-curl",
            "title": "Tomato Yellow Leaf Curl Virus",
            "species": "Tomato",
            "expected_condition": "Yellow Leaf Curl Virus",
            "filename": "tomato_yellow_leaf_curl.jpg",
        },
        {
            "id": "corn-blight",
            "title": "Corn Northern Leaf Blight",
            "species": "Corn",
            "expected_condition": "Northern Leaf Blight",
            "filename": "corn_northern_leaf_blight.jpg",
        },
    ]

    for item in predefined:
        fp = samples_dir / item["filename"]
        if fp.exists():
            samples.append(
                SampleImageItem(
                    id=item["id"],
                    title=item["title"],
                    species=item["species"],
                    expected_condition=item["expected_condition"],
                    filename=item["filename"],
                    url=f"/static/samples/{item['filename']}",
                )
            )

    return samples


@router.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_leaf(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    top_k: int = Form(5),
    auto_crop: bool = Form(True),
    include_visualizations: bool = Form(True),
    target_species: Optional[str] = Form(None),
    use_neural_detector: bool = Form(True),
    selected_box_idx: Optional[int] = Form(None),
):
    """
    End-to-end multi-stage plant leaf diagnosis with two-stage neural detection (YOLOv8)
    and smart leaf ROI isolation. Accepts an uploaded image file OR a remote image URL.
    Optionally accepts a target_species constraint (e.g. Potato, Tomato, Apple) to eliminate cross-crop false positives.
    Optionally accepts selected_box_idx to diagnose a specific candidate leaf in multi-leaf scenes.
    Returns complete diagnosis, severity assessment, treatment recommendations, and visual overlays.
    """
    pipeline = get_pipeline(request)

    if file is None and (not image_url or not image_url.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an image file upload or an image_url must be provided.",
        )

    # Load image into RGB numpy array
    try:
        if file is not None:
            contents = await file.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            img_input = np.array(pil_img)
            filename = file.filename or "uploaded_leaf.jpg"
        else:
            img_input = image_url.strip()
            filename = image_url.strip().split("/")[-1].split("?")[0] or "remote_leaf.jpg"

        # Execute full multi-stage pipeline with YOLO leaf detection and optional species locking
        diagnosis_data = pipeline.diagnose(
            image_input=img_input,
            generate_visualization=include_visualizations,
            top_k=max(1, min(top_k, 10)),
            auto_crop=auto_crop,
            target_species=target_species,
            use_neural_detector=use_neural_detector,
            selected_box_idx=selected_box_idx,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image for diagnosis: {str(exc)}",
        )

    # Encode visualizations if requested
    vis_res = None
    if include_visualizations and "visualizations" in diagnosis_data:
        v = diagnosis_data["visualizations"]
        vis_res = VisualizationsResult(
            detection_overlay=encode_array_to_base64(v.get("detection_overlay")),
            cropped_leaf=encode_array_to_base64(v.get("cropped_leaf")),
            cam_overlay=encode_array_to_base64(v.get("cam_overlay")),
            cam_heatmap=encode_array_to_base64(v.get("cam_heatmap")),
            disease_mask=encode_array_to_base64(v.get("disease_mask")),
            composite_panel=encode_array_to_base64(v.get("composite_panel")),
        )

    # Format detected candidate leaves
    raw_leaves = diagnosis_data.get("detected_leaves", [])
    detected_leaf_items = [
        DetectedLeafItem(
            index=int(item["index"]),
            bbox=list(item["bbox"]),
            confidence=float(item["confidence"]),
            class_id=int(item["class_id"]),
            label=str(item["label"]),
            area=int(item["area"]),
            width=int(item["width"]),
            height=int(item["height"]),
            is_primary=bool(item.get("is_primary", False)),
        )
        for item in raw_leaves
    ]

    return DiagnosisResponse(
        success=True,
        image_name=filename,
        is_cropped=diagnosis_data.get("is_cropped", False),
        roi_bbox=diagnosis_data.get("roi_bbox"),
        detector_used=diagnosis_data.get("detector_used"),
        detected_leaves=detected_leaf_items,
        target_species=diagnosis_data.get("target_species"),
        is_healthy=diagnosis_data["is_healthy"],
        overall_status=diagnosis_data["overall_status"],
        stage1_binary=Stage1Result(**diagnosis_data["stage1_binary"]),
        stage2_fine=Stage2Result(**diagnosis_data["stage2_fine"]),
        stage3_severity=Stage3Result(**diagnosis_data["stage3_severity"]),
        stage4_treatment=Stage4Result(**diagnosis_data["stage4_treatment"]),
        treatment_summary_text=diagnosis_data["treatment_summary_text"],
        visualizations=vis_res,
    )
