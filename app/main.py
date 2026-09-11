"""
LeafDoc Serving Engine & Web Application Entry Point.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from src.inference.pipeline import LeafDocPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager: preloads the multi-stage deep learning pipeline
    and agronomic treatment database once on startup for instant inference.
    """
    print("=" * 60)
    print("Initializing LeafDoc End-to-End Diagnostic Pipeline...")
    print("=" * 60)
    app.state.pipeline = LeafDocPipeline(
        config_path="configs/config.yaml",
        stage1_ckpt_path="checkpoints/stage1_binary_best.pth",
        stage2_ckpt_path="checkpoints/stage2_fine_best.pth",
        treatments_path="data/treatments.json",
    )
    print("LeafDoc Pipeline successfully loaded into app.state.pipeline!")
    print(f"Device: {app.state.pipeline.device}")
    print(f"Classes: {len(app.state.pipeline.s2_idx_to_class)}")
    print("=" * 60)
    yield
    print("Shutting down LeafDoc server.")


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""
    app = FastAPI(
        title="LeafDoc - AI Plant Health & Disease Diagnosis",
        description="Multi-stage Explainable AI Diagnostic System with Quantitative Severity Assessment and Agronomic Prescriptions.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable Cross-Origin Resource Sharing (CORS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files mounting
    static_path = Path("app/static")
    static_path.mkdir(parents=True, exist_ok=True)
    (static_path / "css").mkdir(parents=True, exist_ok=True)
    (static_path / "js").mkdir(parents=True, exist_ok=True)
    (static_path / "samples").mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # API Routes
    app.include_router(api_router)

    # Web Dashboard Root Route
    @app.get("/", response_class=FileResponse)
    async def serve_index():
        index_file = Path("app/templates/index.html")
        if not index_file.exists():
            return HTMLResponse("<h1>LeafDoc</h1><p>Web dashboard is loading...</p>")
        return FileResponse(str(index_file))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
