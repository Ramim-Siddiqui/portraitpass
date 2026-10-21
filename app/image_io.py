import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Settings
from .errors import PipelineRejection, RejectionCode


@dataclass(frozen=True, slots=True)
class LoadedImage:
    rgb: Image.Image
    analysis_rgb: np.ndarray
    filename: str
    width: int
    height: int


def load_image(data: bytes, filename: str, settings: Settings) -> LoadedImage:
    if not data:
        raise PipelineRejection(RejectionCode.EMPTY_FILE)
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise PipelineRejection(RejectionCode.FILE_TOO_LARGE, {"maximum_bytes": max_bytes})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                source_format = (source.format or "").upper()
                if source_format not in {"JPEG", "PNG", "WEBP"}:
                    raise PipelineRejection(RejectionCode.UNSUPPORTED_IMAGE_TYPE,
                                            {"detected_format": source_format or None})
                if bool(getattr(source, "is_animated", False)) or int(getattr(source, "n_frames", 1)) > 1:
                    raise PipelineRejection(RejectionCode.ANIMATED_IMAGE_NOT_ALLOWED)
                width, height = source.size
                if width < settings.min_image_width or height < settings.min_image_height:
                    raise PipelineRejection(RejectionCode.IMAGE_TOO_SMALL,
                                            {"width": width, "height": height,
                                             "minimum_width": settings.min_image_width,
                                             "minimum_height": settings.min_image_height})
                if (width > settings.max_image_width or height > settings.max_image_height
                        or width * height > settings.max_megapixels * 1_000_000):
                    raise PipelineRejection(RejectionCode.IMAGE_TOO_LARGE,
                                            {"width": width, "height": height,
                                             "maximum_pixels": settings.max_megapixels * 1_000_000})
                image = ImageOps.exif_transpose(source).convert("RGB").copy()
    except PipelineRejection:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning):
        raise PipelineRejection(RejectionCode.IMAGE_DECODE_FAILED)
    array = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    return LoadedImage(image, array, filename, image.width, image.height)
