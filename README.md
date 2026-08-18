# LeafDoc — Plant Disease Diagnosis Pipeline

Two-stage deep learning pipeline for plant leaf disease diagnosis on the
PlantVillage dataset (38 classes), with Grad-CAM severity estimation and a
treatment recommendation lookup, served via FastAPI.

```
Upload leaf image
  -> Stage 1: Binary classifier (Healthy vs Diseased, any plant)
  -> Stage 2 (if diseased): Fine-grained classifier (38-class species + disease)
  -> Stage 3: Grad-CAM heatmap on Stage 2 prediction -> % leaf area affected -> Mild/Moderate/Severe
  -> Stage 4: Treatment recommendation lookup table -> advice text
  -> Output: annotated image + diagnosis + severity + recommendation
```

## Stack
PyTorch (EfficientNet-B0 / ResNet-50, pretrained + fine-tuned) · Albumentations ·
pytorch-grad-cam · FastAPI · plain HTML/JS frontend · Docker.
Trained on an RTX 4050 — mixed precision (`torch.cuda.amp`) is on by default in the configs.

---

## Week 1 — you are here

### 1. Environment setup

```bash
cd leafdoc
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify CUDA is visible:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False`, your `requirements.txt` pulled a CPU-only
wheel — reinstall PyTorch from https://pytorch.org/get-started/locally/ with the
CUDA build matching your driver (RTX 4050 → CUDA 12.1+ is fine).

### 2. Get the dataset

Download the PlantVillage dataset (color images, 38 classes) — the most common
mirror is on Kaggle:

```bash
# requires kaggle CLI + API token in ~/.kaggle/kaggle.json
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

If you use a different mirror, just make sure the end result is one folder per
class, each full of images, e.g.:

```
data/raw/
  Apple___Black_rot/
  Apple___healthy/
  Tomato___Late_blight/
  Tomato___healthy/
  ...
```

Update `configs/config.yaml` -> `data.raw_dir` to point at wherever that lands.

### 3. Derive labels + stratified split

```bash
python -m src.data.prepare_dataset --config configs/config.yaml
```

This scans the class folders, derives:
- `fine_label` — the 38-class folder name itself (e.g. `Tomato___Late_blight`)
- `binary_label` — `healthy` / `diseased`, from whether the folder name ends in `healthy`
- `species` — the part before `___`

...then does one **stratified 80/10/10 split by `fine_label`**, so the same
split is automatically consistent for the binary task too (every fine class is
proportionally represented in train/val/test, and since binary is just a
regrouping of fine labels, it inherits the same stratification). Writes
`data/splits/{train,val,test}.csv` with columns `filepath, species, disease,
binary_label, fine_label`.

### 4. Sanity-check with a visualization grid

```bash
python -m src.data.visualize_grid --config configs/config.yaml --split train --n 32
```

Saves `checks/train_grid.png` — a labeled grid of random samples. Actually look
at it before moving on; this is the cheapest bug catch in the whole project
(mislabeled folders, corrupt images, wrong color channels all show up here).

### 5. Augmentation pipeline

`src/data/augmentations.py` defines the Albumentations train/val pipelines
(flip, rotate, brightness/contrast, blur, normalize). Preview it on real images:

```bash
python -m src.data.visualize_grid --config configs/config.yaml --split train --n 32 --augmented
```

---

## What's next (Week 2+)
- `src/models/` — Stage 1 (binary) and Stage 2 (38-class) training scripts, per-class F1 + confusion matrix (not built yet — Week 2)
- `src/inference/gradcam.py` — Grad-CAM severity thresholding (Week 3)
- `app/` — FastAPI backend + HTML/JS frontend + Dockerfile (Week 4)

## Project layout

```
leafdoc/
  configs/config.yaml       # all paths, hyperparams, thresholds in one place
  src/
    data/
      prepare_dataset.py    # label derivation + stratified split
      augmentations.py      # Albumentations train/val pipelines
      visualize_grid.py     # sanity-check grid
      dataset.py            # PyTorch Dataset classes (binary + fine)
    utils/
      seed.py               # reproducibility helper
  data/
    raw/                    # PlantVillage class folders (you populate this)
    splits/                 # generated CSVs
  checks/                   # sanity-check images land here
```
