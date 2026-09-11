# ==============================================================================
# LeafDoc Production Dockerfile
# Multi-stage Explainable Plant Disease Diagnosis & Treatment Prescription API
# ==============================================================================

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install minimal OS dependencies for OpenCV headless and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (for lightweight, highly compatible container deployment)
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application directories & essential files
COPY configs/ ./configs/
COPY checkpoints/ ./checkpoints/
COPY data/treatments.json ./data/treatments.json
COPY data/splits/class_map_*.json ./data/splits/
COPY src/ ./src/
COPY app/ ./app/

# Expose HTTP port
EXPOSE 8000

# Health check to ensure API is responsive
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start FastAPI serving server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
