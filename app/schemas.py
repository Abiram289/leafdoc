"""
Pydantic data models and schemas for LeafDoc FastAPI endpoints.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Stage1Result(BaseModel):
    label: str = Field(..., description="Binary classification label ('healthy' or 'diseased')")
    confidence: float = Field(..., description="Confidence score for Stage 1 prediction (0.0 to 1.0)")
    is_diseased: bool = Field(..., description="True if Stage 1 predicted diseased")
    probabilities: Dict[str, float] = Field(..., description="Softmax probabilities for binary classes")


class TopPrediction(BaseModel):
    label: str = Field(..., description="Fine-grained class name")
    confidence: float = Field(..., description="Confidence score for this class")


class Stage2Result(BaseModel):
    fine_label: str = Field(..., description="Full fine-grained PlantVillage label (e.g. Tomato___Early_blight)")
    species: str = Field(..., description="Parsed plant species name")
    disease: str = Field(..., description="Parsed disease or condition name")
    condition_name: str = Field(..., description="Formatted agronomic condition name")
    confidence: float = Field(..., description="Confidence score for top fine-grained class")
    top_predictions: List[TopPrediction] = Field(default_factory=list, description="Top-k predictions")
    target_species_applied: Optional[str] = Field(None, description="Species filter constraint applied if any")


class Stage3Result(BaseModel):
    severity: str = Field(..., description="Severity category: Healthy, Mild (<15%), Moderate (15-40%), or Severe (>40%)")
    affected_area_pct: float = Field(..., description="Percentage of leaf surface area affected by symptoms")
    total_leaf_pixels: int = Field(..., description="Total number of pixels belonging to segmented leaf")
    affected_pixels: int = Field(..., description="Number of pixels classified as diseased lesions")


class Stage4Result(BaseModel):
    species: str = Field(..., description="Plant species")
    condition_name: str = Field(..., description="Agronomic condition name")
    pathogen_type: str = Field(..., description="Pathogen type (e.g., Fungal, Bacterial, Viral, None)")
    description: str = Field(..., description="Description of symptoms and progression")
    cultural_controls: List[str] = Field(default_factory=list, description="Sanitation and physical cultural practices")
    chemical_controls: List[str] = Field(default_factory=list, description="Approved fungicides / chemical sprays")
    biological_controls: List[str] = Field(default_factory=list, description="Organic and biological remedies")
    preventive_measures: List[str] = Field(default_factory=list, description="Long-term prevention techniques")
    severity_actions: Dict[str, str] = Field(default_factory=dict, description="Actions tailored to severity levels")
    immediate_action: str = Field(..., description="Urgent immediate action for current severity rating")


class DetectedLeafItem(BaseModel):
    index: int = Field(..., description="0-indexed candidate leaf ranking")
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] bounding box coordinates")
    confidence: float = Field(..., description="YOLO detection confidence score (0.0 to 1.0)")
    class_id: int = Field(..., description="Detector class ID")
    label: str = Field(..., description="Detected plant species/leaf label")
    area: int = Field(..., description="Bounding box area in pixels")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")
    is_primary: bool = Field(False, description="True if this candidate leaf was diagnosed")


class VisualizationsResult(BaseModel):
    detection_overlay: Optional[str] = Field(None, description="Base64 encoded JPEG of image with YOLO detection bounding boxes")
    cropped_leaf: Optional[str] = Field(None, description="Base64 encoded JPEG of isolated leaf ROI")
    cam_overlay: Optional[str] = Field(None, description="Base64 encoded JPEG of Grad-CAM overlay on leaf image")
    cam_heatmap: Optional[str] = Field(None, description="Base64 encoded JPEG of raw Grad-CAM activation heatmap")
    disease_mask: Optional[str] = Field(None, description="Base64 encoded PNG of binary lesion segmentation mask")
    composite_panel: Optional[str] = Field(None, description="Base64 encoded JPEG of 4-panel diagnostic dashboard")


class DiagnosisResponse(BaseModel):
    success: bool = True
    image_name: str = Field(..., description="Source image filename or URL")
    is_cropped: bool = Field(False, description="True if smart leaf ROI detection cropped the image")
    roi_bbox: Optional[List[int]] = Field(None, description="Bounding box [x, y, w, h] of cropped leaf")
    detector_used: Optional[str] = Field(None, description="Detection mechanism ('yolov8', 'grabcut', 'grabcut_fallback', or 'none')")
    detected_leaves: List[DetectedLeafItem] = Field(default_factory=list, description="Candidate leaves detected in image")
    target_species: Optional[str] = Field(None, description="Applied target plant species constraint if any")
    is_healthy: bool = Field(..., description="Overall consensus health status flag")
    overall_status: str = Field(..., description="'Healthy' or 'Diseased'")
    stage1_binary: Stage1Result
    stage2_fine: Stage2Result
    stage3_severity: Stage3Result
    stage4_treatment: Stage4Result
    treatment_summary_text: str = Field(..., description="Formatted multi-line text agronomic summary")
    visualizations: Optional[VisualizationsResult] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str = Field(..., description="Compute device in use (cuda or cpu)")
    device_name: Optional[str] = Field(None, description="GPU model name if CUDA is available")
    stage1_checkpoint: str = Field(..., description="Stage 1 model checkpoint status")
    stage2_checkpoint: str = Field(..., description="Stage 2 model checkpoint status")
    total_classes: int = Field(..., description="Number of fine-grained plant disease classes supported")
    total_treatments: int = Field(..., description="Number of treatment entries in knowledgebase")


class ClassItem(BaseModel):
    fine_label: str
    species: str
    condition: str
    is_healthy: bool


class ClassesResponse(BaseModel):
    total_classes: int
    classes: List[ClassItem]
    species_list: List[str]


class SampleImageItem(BaseModel):
    id: str
    title: str
    species: str
    expected_condition: str
    filename: str
    url: str
