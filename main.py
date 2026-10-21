import hmac
import io
import os
import threading
import urllib.request
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import Response
from PIL import Image
import torch

from app.config import Settings
from app.errors import PipelineRejection, RejectionCode
from app.image_io import load_image
from app.remover import BackgroundRemover
from app.validator import PortraitValidator

# Model Download Setup
MODEL_FILES = {
    "models/face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
    "models/pose_landmarker_full.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    ),
}

def ensure_models() -> None:
    for filename, url in MODEL_FILES.items():
        path = Path(filename)
        if path.is_file() and path.stat().st_size > 100_000:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)

ensure_models()

# Load App Configurations & ML Models
settings = Settings()
validator = PortraitValidator(settings)
device = "cuda" if torch.cuda.is_available() else "cpu"
remover = BackgroundRemover(
    settings.model_id,
    settings.model_revision,
    settings.model_input_size,
    settings.torch_threads,
    device=device,
)

pipeline_lock = threading.Lock()

# Initialize FastAPI App
app = FastAPI(title="PortraitPass API")

def verify_api_key(x_api_key: str | None = None):
    # API_KEY may come from the environment or from .env (read by Settings).
    configured = (os.getenv("API_KEY") or settings.api_key or "").strip()
    if configured and not hmac.compare_digest((x_api_key or "").encode(), configured.encode()):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")

@app.get("/health")
def health_check():
    return {"status": "ok", "device": device}

@app.post("/remove-bg")
async def process_portrait_api(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None)
):
    # Verify API key if configured
    verify_api_key(x_api_key)

    if not file:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        data = await file.read()

        # Run CPU validation with thread lock
        with pipeline_lock:
            loaded = load_image(data, file.filename or "uploaded.jpg", settings)
            metrics = validator.validate(loaded)

        # Run Background Removal Model
        processed_image = remover.remove_image(loaded.rgb)

        # Convert output PIL Image to Bytes Stream
        output_buffer = io.BytesIO()
        processed_image.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        # Return transparent PNG directly
        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png",
            headers={
                "X-Validation-Accepted": str(metrics.get("accepted", True)),
                "X-Validation-Status": str(metrics.get("status", "ACCEPTED"))
            }
        )

    except PipelineRejection as exc:
        raise HTTPException(status_code=422, detail=exc.as_dict())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "PROCESSING_FAILED",
                "message": "The photograph could not be processed.",
                "details": {"exception_type": type(exc).__name__, "error": str(exc)},
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)