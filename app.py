import spaces  # MUST be imported first before torch or heavy libraries

import json
import os
import threading
import urllib.request
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from app.config import Settings
from app.errors import PipelineRejection, RejectionCode
from app.image_io import load_image
from app.remover import BackgroundRemover
from app.validator import PortraitValidator


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
settings = Settings()
validator = PortraitValidator(settings)

# Target CUDA dynamically for ZeroGPU
device = "cuda" if torch.cuda.is_available() else "cpu"
remover = BackgroundRemover(
    settings.model_id,
    settings.model_revision,
    settings.model_input_size,
    settings.torch_threads,
    device=device,
)

pipeline_lock = threading.Lock()


def _authorized(request: gr.Request | None = None) -> bool:
    configured = os.getenv("API_KEY", "").strip()
    if not configured:
        return True
    supplied = "" if request is None else request.headers.get("x-api-key", "")
    return supplied == configured


@spaces.GPU(duration=30)
def remove_on_gpu(image: Image.Image) -> Image.Image:
    return remover.remove_image(image)


def process_portrait(image_path: str | None, request: gr.Request | None = None):
    if not _authorized(request):
        issue = PipelineRejection(RejectionCode.UNAUTHORIZED).as_dict()
        return None, issue

    if not image_path:
        issue = PipelineRejection(RejectionCode.EMPTY_FILE).as_dict()
        return None, issue

    try:
        path = Path(image_path)
        data = path.read_bytes()

        # Execute CPU validation under lock
        with pipeline_lock:
            loaded = load_image(data, path.name, settings)
            metrics = validator.validate(loaded)

        # Run inference via @spaces.GPU OUTSIDE thread lock to prevent scheduler deadlocks
        output = remove_on_gpu(loaded.rgb)

        return output, {
            "success": True,
            "accepted": True,
            "status": "ACCEPTED",
            "message": "Portrait accepted and background removed.",
            "image_format": "PNG",
            "transparent_background": True,
            "validation": metrics,
        }
    except PipelineRejection as exc:
        return None, exc.as_dict()
    except Exception as exc:
        return None, {
            "success": False,
            "accepted": False,
            "status": "ERROR",
            "error": {
                "code": "PROCESSING_FAILED",
                "message": "The photograph could not be processed. Please try again.",
                "details": {"exception_type": type(exc).__name__},
            },
        }


with gr.Blocks(title="Student Portrait Background Remover") as demo:
    gr.Markdown(
        "# Student Portrait Background Remover\n"
        "Upload one front-facing, head-and-shoulders or waist-up student portrait. "
        "Accepted images are returned as transparent PNG files."
    )
    with gr.Row():
        source = gr.Image(type="filepath", label="Student portrait")
        result = gr.Image(type="pil", format="png", label="Transparent PNG")
    status = gr.JSON(label="Validation result")
    submit = gr.Button("Validate and remove background", variant="primary")
    submit.click(
        process_portrait,
        inputs=[source],
        outputs=[result, status],
        api_name="process_portrait",
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )