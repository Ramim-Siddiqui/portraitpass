from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"


class RejectionCode(str, Enum):
    EMPTY_FILE = "EMPTY_FILE"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_UPLOAD = "INVALID_UPLOAD"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_IMAGE_TYPE = "UNSUPPORTED_IMAGE_TYPE"
    ANIMATED_IMAGE_NOT_ALLOWED = "ANIMATED_IMAGE_NOT_ALLOWED"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    NO_POSE = "NO_POSE"
    MULTIPLE_PEOPLE = "MULTIPLE_PEOPLE"
    FACE_TOO_FAR = "FACE_TOO_FAR"
    FACE_TOO_CLOSE = "FACE_TOO_CLOSE"
    FACE_CROPPED = "FACE_CROPPED"
    FACE_NOT_CENTERED = "FACE_NOT_CENTERED"
    FACE_NOT_FRONTAL = "FACE_NOT_FRONTAL"
    GAZE_NOT_FORWARD = "GAZE_NOT_FORWARD"
    SHOULDERS_NOT_VISIBLE = "SHOULDERS_NOT_VISIBLE"
    FULL_BODY_VISIBLE = "FULL_BODY_VISIBLE"
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"
    IMAGE_LOW_CONTRAST = "IMAGE_LOW_CONTRAST"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # Deployment-specific codes retained by the Docker API.
    QUEUE_FULL = "QUEUE_FULL"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    PROCESSING_FAILED = "PROCESSING_FAILED"


REJECTION_MESSAGES: dict[RejectionCode, str] = {
    RejectionCode.EMPTY_FILE: "The uploaded file is empty.",
    RejectionCode.FILE_NOT_FOUND: "The selected image could not be found.",
    RejectionCode.INVALID_UPLOAD: "The uploaded file could not be processed.",
    RejectionCode.FILE_TOO_LARGE: "The image file is too large. Upload a file no larger than 2 MB.",
    RejectionCode.UNSUPPORTED_IMAGE_TYPE: "Upload a JPEG, PNG, or WebP image.",
    RejectionCode.ANIMATED_IMAGE_NOT_ALLOWED: "Animated images are not supported. Upload a still photograph.",
    RejectionCode.IMAGE_DECODE_FAILED: "The image is corrupted or could not be decoded.",
    RejectionCode.IMAGE_TOO_SMALL: "The image resolution is too low. Upload a clearer, larger photograph.",
    RejectionCode.IMAGE_TOO_LARGE: "The image dimensions are too large.",
    RejectionCode.NO_FACE: "No clear face was detected. Upload a front-facing portrait.",
    RejectionCode.MULTIPLE_FACES: "More than one face was detected. Upload a photograph containing only one person.",
    RejectionCode.NO_POSE: "The upper body could not be detected clearly.",
    RejectionCode.MULTIPLE_PEOPLE: "More than one person was detected. Upload an individual portrait.",
    RejectionCode.FACE_TOO_FAR: "The person is too far from the camera. Upload a closer portrait.",
    RejectionCode.FACE_TOO_CLOSE: "The face is too close to the camera. Include the head and both shoulders.",
    RejectionCode.FACE_CROPPED: "Part of the face or head is outside the image.",
    RejectionCode.FACE_NOT_CENTERED: "Center the person in the photograph.",
    RejectionCode.FACE_NOT_FRONTAL: "Look directly toward the camera and upload a front-facing photograph.",
    RejectionCode.GAZE_NOT_FORWARD: "Look directly at the camera.",
    RejectionCode.SHOULDERS_NOT_VISIBLE: "Both shoulders must be visible in the photograph.",
    RejectionCode.FULL_BODY_VISIBLE: "The person is too far away. Upload a head-and-shoulders or waist-up photograph.",
    RejectionCode.IMAGE_TOO_BLURRY: "The photograph is too blurry. Upload a sharper image.",
    RejectionCode.IMAGE_TOO_DARK: "The photograph is too dark. Use better lighting.",
    RejectionCode.IMAGE_TOO_BRIGHT: "The photograph is overexposed. Use more balanced lighting.",
    RejectionCode.IMAGE_LOW_CONTRAST: "The face is not sufficiently clear in the photograph.",
    RejectionCode.INTERNAL_ERROR: "The photograph could not be processed. Please try another image.",
    RejectionCode.QUEUE_FULL: "The server is busy. Please try again shortly.",
    RejectionCode.JOB_NOT_FOUND: "The job was not found or has expired.",
    RejectionCode.UNAUTHORIZED: "A valid API key is required.",
    RejectionCode.PROCESSING_FAILED: "The photograph could not be processed. Please try another image.",
}


def rejection_message(code: RejectionCode) -> str:
    return REJECTION_MESSAGES.get(code, "The image does not meet the portrait requirements.")


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: RejectionCode
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code.value, "message": rejection_message(self.code), "details": self.details}


class PipelineRejection(Exception):
    def __init__(self, code: RejectionCode | str, details: dict[str, Any] | None = None,
                 issues: list[ValidationIssue] | None = None, status: ValidationStatus = ValidationStatus.REJECTED):
        self.code = code if isinstance(code, RejectionCode) else RejectionCode(code)
        self.details = details or {}
        self.issues = issues or [ValidationIssue(self.code, self.details)]
        self.status = status
        super().__init__(rejection_message(self.code))

    @property
    def message(self) -> str:
        return rejection_message(self.code)

    @property
    def http_status(self) -> int:
        if self.code == RejectionCode.QUEUE_FULL:
            return 503
        if self.code == RejectionCode.JOB_NOT_FOUND:
            return 404
        if self.code == RejectionCode.UNAUTHORIZED:
            return 401
        if self.code == RejectionCode.FILE_TOO_LARGE:
            return 413
        if self.status == ValidationStatus.ERROR or self.code in {
            RejectionCode.INTERNAL_ERROR, RejectionCode.PROCESSING_FAILED
        }:
            return 500
        if self.code in {
            RejectionCode.EMPTY_FILE, RejectionCode.FILE_NOT_FOUND, RejectionCode.INVALID_UPLOAD,
            RejectionCode.UNSUPPORTED_IMAGE_TYPE, RejectionCode.ANIMATED_IMAGE_NOT_ALLOWED,
            RejectionCode.IMAGE_DECODE_FAILED, RejectionCode.IMAGE_TOO_SMALL, RejectionCode.IMAGE_TOO_LARGE,
        }:
            return 400
        return 422

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": False,
            "accepted": False,
            "status": self.status.value,
            "error": {"code": self.code.value, "message": self.message, "details": self.details},
            "rejection_reasons": [issue.as_dict() for issue in self.issues],
        }
