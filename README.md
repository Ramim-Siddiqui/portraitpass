# PortraitPass

Checks student ID photos and removes their background. It runs as a FastAPI service for portals and as a Gradio demo on Hugging Face.

**Live demo:** [huggingface.co/spaces/ramimali/student-portrait-bg-remover](https://huggingface.co/spaces/ramimali/student-portrait-bg-remover)

Every upload is validated against a portrait policy: one person, a face the camera can see, a head turned toward the camera, no full-body shots and no blur. An accepted photo comes back as a transparent PNG cut out with BiRefNet Portrait. A rejected photo comes back with machine-readable codes and plain-English messages, so the portal that called the service can tell the student exactly what to fix.

**Built with:** FastAPI, Gradio, MediaPipe Face Landmarker and Pose Landmarker, BiRefNet Portrait (Hugging Face Transformers, PyTorch), OpenCV, Pillow, Docker, Hugging Face Spaces (ZeroGPU).

## How it works

1. **Upload checks** (`app/image_io.py`). JPEG, PNG or WebP only, at most 2 MB, between 200 and 12,000 pixels a side and at most 40 megapixels. Animated images and decompression bombs are refused, and the EXIF orientation is applied before anything else.
2. **Portrait policy** (`app/validator.py`).
   - MediaPipe's face and pose landmarkers scan the whole image and a grid of overlapping tiles, so a smaller person in the background is still found. Candidates from different tiles are merged by box overlap and centre distance.
   - Exactly one face and one person are allowed.
   - Head yaw, pitch and roll, taken from the face landmarker's transformation matrix, must each be within 25 degrees.
   - A photo where the knees or ankles are visible is a full-body shot and is rejected.
   - Sharpness is the variance of the Laplacian; below 80 the photo is too blurry. Brightness and contrast are measured and reported but do not reject.
3. **Background removal** (`app/remover.py`). BiRefNet Portrait predicts a mask at 1024 x 1024. The mask is resized to the original photo, lightly cleaned of isolated noise and feathered so hair and soft edges survive, then applied as the alpha channel.

Uploaded and generated images are processed in memory and never stored.

## API

| Endpoint | Body | Returns |
| --- | --- | --- |
| `GET /health` | | `{"status": "ok", "device": "cpu"}` |
| `POST /remove-bg` | multipart `file` | `200`: the transparent PNG, with `X-Validation-Accepted` and `X-Validation-Status` headers. `422`: the rejection (below). `401`: missing or wrong API key. `500`: `PROCESSING_FAILED`. |

When `API_KEY` is set, every `POST` must send the same value in the `X-API-Key` header.

A rejection lists every reason found, with the first one as the main error:

```json
{
  "detail": {
    "success": false,
    "accepted": false,
    "status": "REJECTED",
    "error": {
      "code": "NO_FACE",
      "message": "No clear face was detected. Upload a front-facing portrait.",
      "details": {"detected_face_count": 0}
    },
    "rejection_reasons": [
      {"code": "NO_FACE", "message": "No clear face was detected. Upload a front-facing portrait.", "details": {"detected_face_count": 0}},
      {"code": "IMAGE_TOO_BLURRY", "message": "The photograph is too blurry. Upload a sharper image.", "details": {"blur_score": 0.0, "minimum_blur_score": 80.0}}
    ]
  }
}
```

The codes are defined in `app/errors.py`. The ones the policy uses most are `NO_FACE`, `MULTIPLE_FACES`, `MULTIPLE_PEOPLE`, `FACE_NOT_FRONTAL`, `FULL_BODY_VISIBLE`, `IMAGE_TOO_BLURRY`, `FILE_TOO_LARGE`, `UNSUPPORTED_IMAGE_TYPE` and `IMAGE_DECODE_FAILED`.

## Run it locally

Python 3.10 or 3.11 (MediaPipe 0.10.14 does not support newer versions).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- The first start downloads the two MediaPipe models (about 13 MB) into `models/` and BiRefNet Portrait (about 850 MB) into the Hugging Face cache.
- On a CPU one photo takes roughly 40 seconds, most of it in BiRefNet at 1024 pixels. A CUDA GPU is used automatically when PyTorch can see one.
- Try it at http://127.0.0.1:8000/docs, or:

```powershell
curl.exe -F "file=@photo.jpg" http://127.0.0.1:8000/remove-bg -o cutout.png
```

## The Hugging Face demo

`app.py` is the Gradio front end that runs on Hugging Face Spaces. It uses the same validation and background-removal code, and runs BiRefNet on a ZeroGPU GPU through the `spaces` package. Its API name is `/process_portrait`; the Space's **Use via API** panel shows the request format. To run it locally:

```powershell
.\.venv\Scripts\python.exe -m pip install gradio spaces
.\.venv\Scripts\python.exe app.py
```

## Run it with Docker

```powershell
docker build -t student-portrait .
docker run -p 127.0.0.1:8000:8000 -e API_KEY=change-me student-portrait
```

## Settings

Settings come from environment variables or a `.env` file (see `app/config.py` for all of them).

| Setting | Default | Purpose |
| --- | --- | --- |
| `API_KEY` | (empty: no check) | Required `X-API-Key` header for `POST /remove-bg`. |
| `MODEL_INPUT_SIZE` | `1024` | BiRefNet input size. Smaller is faster but loses fine edges. |
| `TORCH_THREADS` | `2` | CPU threads for PyTorch. |
| `MAX_UPLOAD_MB` | `2` | Largest accepted file. |
| `MAX_ABS_YAW_DEGREES`, `MAX_ABS_PITCH_DEGREES`, `MAX_ABS_ROLL_DEGREES` | `25` | How far the head may turn, nod or tilt. |
| `MINIMUM_BLUR_SCORE` | `80` | Sharpness threshold (variance of the Laplacian). |
| `REJECT_VISIBLE_KNEES`, `REJECT_VISIBLE_ANKLES` | `true` | Full-body detection. |
| `PERSON_SCAN_ROWS`, `PERSON_SCAN_COLS` | `2`, `2` | Tile grid for finding a second person. |

## Files

```
main.py            HTTP API, model download, API key
app.py             Gradio demo for Hugging Face Spaces
app/image_io.py    upload checks and decoding
app/validator.py   face, pose, framing and quality checks
app/remover.py     BiRefNet background removal
app/errors.py      rejection codes and messages
app/config.py      settings
Dockerfile         container image
```

## License

Copyright (c) 2026 Ramim Ali Siddiqui. All rights reserved. See [LICENSE](LICENSE).

BiRefNet Portrait (ZhengPeng7/BiRefNet-portrait) and the MediaPipe models are downloaded at run time under their own licenses.
