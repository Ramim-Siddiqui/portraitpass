from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = ""

    # Background removal
    model_id: str = "ZhengPeng7/BiRefNet-portrait"
    model_revision: str = "ecdeb6240ef23557dbd48ff27c59c1a88cbcb755"
    model_input_size: int = 1024
    torch_threads: int = 2

    # ------------------------------------------------------------------
    # Head orientation: remains a hard rejection rule.
    # ------------------------------------------------------------------
    max_abs_yaw_degrees: float = 25.0
    max_abs_pitch_degrees: float = 25.0
    max_abs_roll_degrees: float = 25.0

    # ------------------------------------------------------------------
    # Relaxed framing policy.
    # These values are retained for compatibility but are NOT rejection
    # criteria in the validator.
    # ------------------------------------------------------------------
    min_face_height_ratio: float = 0.0
    max_face_height_ratio: float = 1.0
    min_face_width_pixels: int = 0
    max_horizontal_center_offset: float = 1.0
    minimum_face_border_margin_ratio: float = 0.0
    minimum_shoulder_visibility: float = 0.0
    require_both_shoulders: bool = False

    # Full-body rejection, only when a reliable pose is available.
    lower_body_visibility_threshold: float = 0.60
    reject_visible_knees: bool = True
    reject_visible_ankles: bool = True

    # Blur remains a hard rejection. Lighting/contrast are metrics only.
    minimum_blur_score: float = 80.0
    minimum_brightness: float = 0.0
    maximum_brightness: float = 255.0
    minimum_contrast: float = 0.0

    # ------------------------------------------------------------------
    # Image/application limits
    # ------------------------------------------------------------------
    queue_capacity: int = 10
    job_ttl_seconds: int = 900
    sync_timeout_seconds: int = 240
    face_model_path: str = "models/face_landmarker.task"
    pose_model_path: str = "models/pose_landmarker_full.task"
    inference_maximum_side: int = 1600

    max_upload_mb: int = 2
    max_megapixels: int = 40
    min_image_width: int = 200
    min_image_height: int = 200
    max_image_width: int = 12000
    max_image_height: int = 12000

    # ------------------------------------------------------------------
    # Secondary-person detection
    # ------------------------------------------------------------------
    # MediaPipe is used instead of a generic YOLO box detector here.
    # We scan the full image plus overlapping tiles and deduplicate the
    # resulting face/pose candidates. This greatly reduces object-shaped
    # false positives while still catching a smaller/background person.
    person_scan_enabled: bool = True
    person_scan_rows: int = 2
    person_scan_cols: int = 2
    person_scan_overlap: float = 0.20
    person_face_detection_confidence: float = 0.35
    person_face_presence_confidence: float = 0.35
    person_pose_detection_confidence: float = 0.30
    person_pose_presence_confidence: float = 0.30
    person_pose_tracking_confidence: float = 0.30
    person_min_visible_landmarks: int = 6
    person_dedup_iou_threshold: float = 0.20
    person_dedup_center_threshold: float = 0.12


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()