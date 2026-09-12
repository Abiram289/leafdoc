# LeafDoc — Explainable Agronomic AI & Plant Disease Diagnosis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LeafDoc** is an end-to-end agronomic AI plant health and disease diagnosis system. Trained on 54,304 images across 38 PlantVillage classes (14 crop species), it pairs dual-stage deep learning with Grad-CAM visual explainability, quantitative lesion severity calculation, and customized agronomic prescriptions.

---

## System Architecture

```
                       [ Leaf Photo / Mobile Capture / Web URL ]
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Stage 0: YOLOv8 Neural Detector    │
                      │  • Specialized 21.5 MB Leaf Model     │
                      │  • Multi-Leaf Candidate Ranking       │
                      │  • Aspect-Ratio Preserving Letterbox  │
                      │  • Fallback to Saliency Preprocessor  │
                      └───────────────────┬───────────────────┘
                                          │
                               (Selected Leaf ROI)
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Stage 1: Binary Health Classifier   │
                      │      (Healthy vs Diseased Leaf)       │
                      │      EfficientNet-B0 [99.95% F1]      │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │  Stage 2: Fine-Grained Identification │
                      │      (38 Classes across 14 Crops)     │
                      │   Optional: Plant Species Filter      │
                      │     EfficientNet-B0 [99.90% Acc]      │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Stage 3: Explainability & Severity  │
                      │  • Grad-CAM Heatmap Localization      │
                      │  • Leaf Mask Morphological Extraction │
                      │  • % Affected Surface Area & Category │
                      │    (Healthy, Mild, Moderate, Severe)  │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │   Stage 4: Agronomic Prescriptions    │
                      │  • Immediate Severity Action Plan     │
                      │  • Cultural Sanitation Measures       │
                      │  • Chemical & Biological Fungicides   │
                      │  • Long-Term Preventative Protocols   │
                      └───────────────────┬───────────────────┘
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    FastAPI REST Engine & UI Client    │
                      │  • Botanical Glassmorphic Dashboard   │
                      │  • Interactive Multi-Leaf Tray (YOLO) │
                      │  • Live Webcam & Drag-and-Drop        │
                      │  • 1-Click Interactive Demo Samples   │
                      │  • Printable Diagnostic PDF Reports   │
                      └───────────────────────────────────────┘
```

---

## Key Features

- **Two-Stage Deep Learning Detection (YOLOv8)**: Stage 0 neural object detector isolates individual leaf contours in complex outdoor and orchard environments, with multi-leaf ranking and interactive candidate leaf selection.
- **Dual-Stage Deep Learning Classification**: Binary screening followed by 38-class fine identification eliminates false positives on vigorous foliage.
- **Smart Saliency Fallback**: Automatic GrabCut & vegetation chrominance fallback for macro close-ups where whole leaf boundaries are absent.
- **Aspect-Ratio Preserving Letterboxing**: Prevents spatial distortion of leaf serrations and vein patterns.
- **Crop Filter Option**: Option to lock diagnosis to specific crops (Potato, Tomato, Apple, Grape, Corn, Peach, Pepper, etc.), re-normalizing probabilities to eliminate cross-crop confusion.
- **Explainable AI (Grad-CAM)**: Backpropagates gradients into `conv_head` to highlight exact lesion biomarkers.
- **Quantitative Severity Meter**: Computes exact percentage of affected leaf tissue with automated categorization (<15% Mild, 15–40% Moderate, >40% Severe).
- **Comprehensive Treatment Encyclopedia**: Curated agronomic database providing chemical, biological, and cultural recommendations for all 38 conditions.

---

## 14 Supported Crop Species (38 Classes)

1. **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
2. **Blueberry**: Healthy
3. **Cherry**: Powdery Mildew, Healthy
4. **Corn (Maize)**: Cercospora Leaf Spot (Gray Leaf Spot), Common Rust, Northern Leaf Blight, Healthy
5. **Grape**: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
6. **Orange**: Huanglongbing (Citrus Greening)
7. **Peach**: Bacterial Spot, Healthy
8. **Pepper (Bell)**: Bacterial Spot, Healthy
9. **Potato**: Early Blight, Late Blight, Healthy
10. **Raspberry**: Healthy
11. **Soybean**: Healthy
12. **Squash**: Powdery Mildew
13. **Strawberry**: Leaf Scorch, Healthy
14. **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted Spider Mite), Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy

---

## Quick Start

### 1. Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/Abiram289/leafdoc.git
cd leafdoc

# Create virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Serving Web Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **`http://localhost:8000`** to access the interactive web dashboard.
Interactive OpenAPI Swagger docs are available at **`http://localhost:8000/docs`**.

### 3. Command-Line Inference

```bash
# Full 4-stage diagnosis on a local image
python -m src.inference.predict --config configs/config.yaml --stage pipeline --image path/to/leaf.jpg

# Full 4-stage diagnosis directly on a remote web URL
python -m src.inference.predict --config configs/config.yaml --stage pipeline --image https://example.com/plant-leaf.jpg
```

---

## Docker Deployment

### Run with Docker Compose:

```bash
docker-compose up --build
```

### Run with Docker:

```bash
docker build -t leafdoc-ai .
docker run -p 8000:8000 leafdoc-ai
```

---

## REST API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/diagnose` | `POST` | Complete 4-stage diagnosis with Grad-CAM overlays and treatments |
| `/api/v1/health` | `GET` | Hardware acceleration and checkpoint readiness health check |
| `/api/v1/classes` | `GET` | List all 38 fine-grained plant species and disease classes |
| `/api/v1/treatments` | `GET` | Agronomic treatment encyclopedia, optionally filtered by species |
| `/api/v1/samples` | `GET` | Curated sample images for instant 1-click UI diagnosis |

---

## Testing & Verification

```bash
python run_tests.py
```

Runs the complete 37-test suite covering:
- Binary and Fine-Grained Model Factories
- Data Augmentation and Stratification Pipelines
- Grad-CAM Heatmap Extraction and Saliency Thresholds
- Saliency ROI Preprocessor and Aspect-Ratio Letterboxer
- Plant Species Filtering and Re-normalization
- Agronomic Treatments Encyclopedia Integrity
- FastAPI REST Endpoints & HTML Dashboard

---

## Repository Layout

```
leafdoc/
├── app/
│   ├── api/routes.py            # FastAPI REST routes
│   ├── static/                  # Glassmorphic CSS, JS client, demo samples
│   ├── templates/index.html     # Web application dashboard
│   ├── main.py                  # ASGI server entry point
│   └── schemas.py               # Pydantic request/response schemas
├── checkpoints/                 # Trained PyTorch model weights (.pth)
├── configs/
│   └── config.yaml              # Hyperparameters, architecture, and thresholds
├── data/
│   ├── splits/                  # Stratified class mappings
│   └── treatments.json          # 38-class agronomic treatments database
├── notebooks/
│   └── leafdoc_demonstration.ipynb # End-to-end interactive demo
├── src/
│   ├── data/                    # Dataset parsing, augmentations, and loaders
│   ├── inference/
│   │   ├── gradcam.py           # Grad-CAM explainability & severity engine
│   │   ├── pipeline.py          # Unified 4-stage inference pipeline
│   │   ├── predict.py           # CLI inference runner
│   │   ├── preprocessor.py      # Smart Leaf ROI & Letterboxing
│   │   └── treatments.py        # Agronomic recommendation lookup
│   ├── models/                  # PyTorch model factory & classifier heads
│   ├── training/                # Dual-stage training engine with AMP
│   └── utils/                   # Config and seed utilities
├── tests/                       # Complete unit and integration test suite
├── Dockerfile                   # Container build definition
├── docker-compose.yml           # Compose orchestration
├── requirements.txt             # Python dependencies
└── run_tests.py                 # Automated test runner
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
