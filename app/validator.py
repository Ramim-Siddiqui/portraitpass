import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .config import Settings
from .errors import PipelineRejection, RejectionCode, ValidationIssue, ValidationStatus
from .image_io import LoadedImage


@dataclass(slots=True)
class FaceCandidate:
    points: np.ndarray
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1, normalized full-image coords
    source: str
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None


@dataclass(slots=True)
class PoseCandidate:
    points: np.ndarray
    bbox: tuple[float, float, float, float]
    source: str


@dataclass(slots=True)
class FaceAnalysis:
    face_count: int = 0
    face_width_pixels: float | None = None
    face_height_pixels: float | None = None
    face_width_ratio: float | None = None
    face_height_ratio: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    face_cropped: bool | None = None
    face_centered: bool | None = None
    frontal: bool | None = None
    landmarks: np.ndarray | None = None


@dataclass(slots=True)
class PoseAnalysis:
    pose_count: int = 0
    left_shoulder_visibility: float | None = None
    right_shoulder_visibility: float | None = None
    knees_visible: bool | None = None
    ankles_visible: bool | None = None
    shoulders_visible: bool | None = None
    full_body_visible: bool | None = None
    landmarks: np.ndarray | None = None


@dataclass(slots=True)
class QualityAnalysis:
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    blurry: bool = False
    too_dark: bool = False
    too_bright: bool = False
    low_contrast: bool = False


def _finite(value, default=0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _landmarks_array(landmarks, confidence: bool) -> np.ndarray:
    rows = []
    for point in landmarks:
        row = [_finite(point.x), _finite(point.y), _finite(point.z)]
        if confidence:
            row += [
                _finite(getattr(point, "visibility", 0.0)),
                _finite(getattr(point, "presence", 0.0)),
            ]
        rows.append(row)
    width = 5 if confidence else 3
    return np.asarray(rows, dtype=np.float32).reshape(-1, width)


def _bbox_from_points(points: np.ndarray) -> tuple[float, float, float, float]:
    visible = points[np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])]
    if visible.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.clip(np.min(visible[:, 0]), 0.0, 1.0)),
        float(np.clip(np.min(visible[:, 1]), 0.0, 1.0)),
        float(np.clip(np.max(visible[:, 0]), 0.0, 1.0)),
        float(np.clip(np.max(visible[:, 1]), 0.0, 1.0)),
    )


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-9 else 0.0


def _bbox_center(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def _boxes_same_person(a, b, iou_threshold: float, center_threshold: float) -> bool:
    if _bbox_iou(a, b) >= iou_threshold:
        return True
    ca, cb = _bbox_center(a), _bbox_center(b)
    aw = max(1e-6, a[2] - a[0])
    ah = max(1e-6, a[3] - a[1])
    bw = max(1e-6, b[2] - b[0])
    bh = max(1e-6, b[3] - b[1])
    scale = max(aw, ah, bw, bh)
    return float(np.linalg.norm(ca - cb)) <= max(center_threshold, 0.6 * scale)


def _resize_for_inference(rgb: np.ndarray, maximum_side: int) -> np.ndarray:
    height, width = rgb.shape[:2]
    largest = max(height, width)
    if largest <= maximum_side:
        return np.ascontiguousarray(rgb).copy()
    scale = maximum_side / largest
    return np.ascontiguousarray(
        cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    )


def _crop_tiles(rgb: np.ndarray, rows: int, cols: int, overlap: float):
    h, w = rgb.shape[:2]
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    overlap = float(np.clip(overlap, 0.0, 0.45))

    tile_w = int(round(w / (cols - overlap * (cols - 1))))
    tile_h = int(round(h / (rows - overlap * (rows - 1))))
    step_x = max(1, int(round(tile_w * (1.0 - overlap))))
    step_y = max(1, int(round(tile_h * (1.0 - overlap))))

    starts_x = [min(max(0, w - tile_w), c * step_x) for c in range(cols)]
    starts_y = [min(max(0, h - tile_h), r * step_y) for r in range(rows)]

    seen = set()
    for sy in starts_y:
        for sx in starts_x:
            key = (sx, sy, min(w, sx + tile_w), min(h, sy + tile_h))
            if key in seen:
                continue
            seen.add(key)
            x1, y1, x2, y2 = key
            tile = rgb[y1:y2, x1:x2]
            yield tile, (x1 / w, y1 / h, x2 / w, y2 / h)


def _map_points_from_tile(points: np.ndarray, tile_box) -> np.ndarray:
    tx0, ty0, tx1, ty1 = tile_box
    out = points.copy()
    out[:, 0] = tx0 + points[:, 0] * (tx1 - tx0)
    out[:, 1] = ty0 + points[:, 1] * (ty1 - ty0)
    return out


def _euler_degrees(transformation_matrix) -> tuple[float, float, float]:
    matrix = np.asarray(transformation_matrix, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError(f"Expected 16 matrix values, received {matrix.size}.")
    rotation = matrix.reshape(4, 4)[:3, :3]
    if not np.isfinite(rotation).all():
        raise ValueError("Invalid face transformation matrix.")
    u_matrix, _, v_transpose = np.linalg.svd(rotation)
    rotation = u_matrix @ v_transpose
    if np.linalg.det(rotation) < 0:
        u_matrix[:, -1] *= -1
        rotation = u_matrix @ v_transpose
    singularity = math.hypot(rotation[0, 0], rotation[1, 0])
    if singularity > 1e-6:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], singularity)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], singularity)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


class PortraitValidator:
    def __init__(self, settings: Settings):
        self.s = settings
        face_path = Path(settings.face_model_path)
        pose_path = Path(settings.pose_model_path)
        if not face_path.is_file() or not pose_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe Task models missing: {face_path}, {pose_path}"
            )

        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions

        self.face_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=base(model_asset_path=str(face_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=5,
                min_face_detection_confidence=settings.person_face_detection_confidence,
                min_face_presence_confidence=settings.person_face_presence_confidence,
                min_tracking_confidence=0.30,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
        )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(pose_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=5,
                min_pose_detection_confidence=settings.person_pose_detection_confidence,
                min_pose_presence_confidence=settings.person_pose_presence_confidence,
                min_tracking_confidence=settings.person_pose_tracking_confidence,
                output_segmentation_masks=False,
            )
        )

    def close(self) -> None:
        self.face_landmarker.close()
        self.pose_landmarker.close()

    def _detect_face_candidates(self, rgb: np.ndarray) -> list[FaceCandidate]:
        candidates: list[FaceCandidate] = []

        def scan(image: np.ndarray, tile_box, source: str):
            media = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
            result = self.face_landmarker.detect(media)
            faces = list(getattr(result, "face_landmarks", []) or [])
            for index, face in enumerate(faces):
                pts = self._landmarks_array(face, False)
                if tile_box is not None:
                    pts = _map_points_from_tile(pts, tile_box)
                yaw = pitch = roll = None
                matrices = list(getattr(result, "facial_transformation_matrixes", []) or [])
                if index < len(matrices):
                    try:
                        yaw, pitch, roll = _euler_degrees(matrices[index])
                    except (ValueError, TypeError, np.linalg.LinAlgError):
                        yaw = pitch = roll = None
                bbox = _bbox_from_points(pts)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    candidates.append(FaceCandidate(pts, bbox, f"{source}:{index}", yaw, pitch, roll))

        scan(rgb, None, "full")

        if self.s.person_scan_enabled:
            for tile, tile_box in _crop_tiles(
                rgb,
                self.s.person_scan_rows,
                self.s.person_scan_cols,
                self.s.person_scan_overlap,
            ):
                scan(tile, tile_box, "tile")

        unique: list[FaceCandidate] = []
        for candidate in sorted(
            candidates,
            key=lambda c: _bbox_area(c.bbox),
            reverse=True,
        ):
            if any(
                _boxes_same_person(
                    candidate.bbox,
                    kept.bbox,
                    self.s.person_dedup_iou_threshold,
                    self.s.person_dedup_center_threshold,
                )
                for kept in unique
            ):
                continue
            unique.append(candidate)

        return unique

    def _detect_pose_candidates(self, rgb: np.ndarray) -> list[PoseCandidate]:
        candidates: list[PoseCandidate] = []

        def scan(image: np.ndarray, tile_box, source: str):
            media = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
            result = self.pose_landmarker.detect(media)
            poses = list(getattr(result, "pose_landmarks", []) or [])
            for index, pose in enumerate(poses):
                pts = self._landmarks_array(pose, True)
                if tile_box is not None:
                    pts = _map_points_from_tile(pts, tile_box)
                visible = int(
                    np.count_nonzero(
                        (pts[:, 3] >= self.s.person_pose_detection_confidence)
                        & (pts[:, 4] >= self.s.person_pose_presence_confidence)
                    )
                )
                if visible < self.s.person_min_visible_landmarks:
                    continue
                bbox = _bbox_from_points(pts)
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    candidates.append(PoseCandidate(pts, bbox, f"{source}:{index}"))

        scan(rgb, None, "full")

        if self.s.person_scan_enabled:
            for tile, tile_box in _crop_tiles(
                rgb,
                self.s.person_scan_rows,
                self.s.person_scan_cols,
                self.s.person_scan_overlap,
            ):
                scan(tile, tile_box, "tile")

        unique: list[PoseCandidate] = []
        candidates.sort(
            key=lambda c: _bbox_area(c.bbox),
            reverse=True,
        )
        for candidate in candidates:
            if any(
                _boxes_same_person(
                    candidate.bbox,
                    kept.bbox,
                    self.s.person_dedup_iou_threshold,
                    self.s.person_dedup_center_threshold,
                )
                for kept in unique
            ):
                continue
            unique.append(candidate)

        return unique

    @staticmethod
    def _landmarks_array(landmarks, confidence: bool) -> np.ndarray:
        return _landmarks_array(landmarks, confidence)

    def _build_face_analysis(self, candidate: FaceCandidate, all_faces: list[FaceCandidate], width: int, height: int) -> FaceAnalysis:
        points = candidate.points
        x0, y0, x1, y1 = candidate.bbox
        fw, fh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        centered = abs(((x0 + x1) / 2.0) - 0.5) <= self.s.max_horizontal_center_offset
        margin = self.s.minimum_face_border_margin_ratio
        cropped = x0 <= margin or y0 <= margin or x1 >= 1.0 - margin or y1 >= 1.0 - margin

        return FaceAnalysis(
            face_count=len(all_faces),
            face_width_pixels=fw * width,
            face_height_pixels=fh * height,
            face_width_ratio=fw,
            face_height_ratio=fh,
            face_cropped=cropped,
            face_centered=centered,
            landmarks=points,
        )

    def _primary_face_index(self, faces: list[FaceCandidate]) -> int:
        if len(faces) == 1:
            return 0
        scores = []
        for face in faces:
            area = _bbox_area(face.bbox)
            center = _bbox_center(face.bbox)
            centrality = max(0.0, 1.0 - float(np.linalg.norm(center - np.array([0.5, 0.5]))) * 1.5)
            scores.append(area * (0.5 + 0.5 * centrality))
        return int(np.argmax(scores))

    def _apply_orientation(self, face: FaceAnalysis, candidate: FaceCandidate) -> FaceAnalysis:
        face.yaw, face.pitch, face.roll = candidate.yaw, candidate.pitch, candidate.roll
        if candidate.yaw is None or candidate.pitch is None or candidate.roll is None:
            face.frontal = None
        else:
            face.frontal = (
                abs(candidate.yaw) <= self.s.max_abs_yaw_degrees
                and abs(candidate.pitch) <= self.s.max_abs_pitch_degrees
                and abs(candidate.roll) <= self.s.max_abs_roll_degrees
            )
        return face

    @staticmethod
    def _confidence(points: np.ndarray, index: int) -> float:
        if index >= len(points):
            return 0.0
        return min(float(points[index, 3]), float(points[index, 4]))

    def _visible(self, points: np.ndarray, index: int, threshold: float) -> bool:
        if index >= len(points):
            return False
        x, y = float(points[index, 0]), float(points[index, 1])
        return -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05 and self._confidence(points, index) >= threshold

    def _pose_analysis(self, poses: list[PoseCandidate], primary_face: FaceAnalysis) -> PoseAnalysis:
        if not poses:
            return PoseAnalysis()

        face_center = _bbox_center((
            float(primary_face.landmarks[:, 0].min()),
            float(primary_face.landmarks[:, 1].min()),
            float(primary_face.landmarks[:, 0].max()),
            float(primary_face.landmarks[:, 1].max()),
        )) if primary_face.landmarks is not None else np.array([0.5, 0.5])

        def distance(pose: PoseCandidate) -> float:
            body_center = _bbox_center(pose.bbox)
            return float(np.linalg.norm(body_center - face_center))

        primary = min(poses, key=distance)
        p = primary.points

        ls = self._confidence(p, 11)
        rs = self._confidence(p, 12)
        left_s = self._visible(p, 11, self.s.minimum_shoulder_visibility)
        right_s = self._visible(p, 12, self.s.minimum_shoulder_visibility)
        shoulders = left_s and right_s if self.s.require_both_shoulders else left_s or right_s
        knees = (
            self._visible(p, 25, self.s.lower_body_visibility_threshold)
            and self._visible(p, 26, self.s.lower_body_visibility_threshold)
        )
        ankles = (
            self._visible(p, 27, self.s.lower_body_visibility_threshold)
            and self._visible(p, 28, self.s.lower_body_visibility_threshold)
        )
        full = (self.s.reject_visible_knees and knees) or (self.s.reject_visible_ankles and ankles)

        return PoseAnalysis(
            pose_count=len(poses),
            left_shoulder_visibility=ls,
            right_shoulder_visibility=rs,
            knees_visible=knees,
            ankles_visible=ankles,
            shoulders_visible=shoulders,
            full_body_visible=bool(full),
            landmarks=p,
        )

    def _quality(self, loaded: LoadedImage, face: FaceAnalysis) -> QualityAnalysis:
        rgb = loaded.analysis_rgb
        h, w = rgb.shape[:2]
        if face.landmarks is not None and len(face.landmarks):
            x0 = float(face.landmarks[:, 0].min() * w)
            x1 = float(face.landmarks[:, 0].max() * w)
            y0 = float(face.landmarks[:, 1].min() * h)
            y1 = float(face.landmarks[:, 1].max() * h)
            ex, ey = (x1 - x0) * 0.20, (y1 - y0) * 0.20
            crop = rgb[
                max(0, round(y0 - ey)): min(h, round(y1 + ey)),
                max(0, round(x0 - ex)): min(w, round(x1 + ex)),
            ]
        else:
            crop = rgb

        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            crop = rgb

        interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > 256 else cv2.INTER_CUBIC
        gray = cv2.cvtColor(cv2.resize(crop, (256, 256), interpolation=interpolation), cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        return QualityAnalysis(
            blur_score=blur,
            brightness=brightness,
            contrast=contrast,
            blurry=blur < self.s.minimum_blur_score,
            too_dark=False,
            too_bright=False,
            low_contrast=False,
        )

    def _issues(self, face: FaceAnalysis, pose: PoseAnalysis, quality: QualityAnalysis, face_scan_count: int, secondary_pose_count: int) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        def add(code: RejectionCode, details=None):
            if not any(item.code == code for item in issues):
                issues.append(ValidationIssue(code, details or {}))

        if face_scan_count == 0:
            add(RejectionCode.NO_FACE, {"detected_face_count": 0})
        elif face_scan_count > 1:
            add(
                RejectionCode.MULTIPLE_FACES,
                {"detected_face_count": face_scan_count, "required_face_count": 1},
            )
        else:
            if face.frontal is None:
                add(
                    RejectionCode.INTERNAL_ERROR,
                    {"stage": "head_pose", "reason": "Head orientation could not be calculated."},
                )
            elif not face.frontal:
                add(
                    RejectionCode.FACE_NOT_FRONTAL,
                    {
                        "yaw_degrees": None if face.yaw is None else round(face.yaw, 2),
                        "pitch_degrees": None if face.pitch is None else round(face.pitch, 2),
                        "roll_degrees": None if face.roll is None else round(face.roll, 2),
                        "maximum_absolute_yaw": self.s.max_abs_yaw_degrees,
                        "maximum_absolute_pitch": self.s.max_abs_pitch_degrees,
                        "maximum_absolute_roll": self.s.max_abs_roll_degrees,
                    },
                )

        if face_scan_count <= 1 and secondary_pose_count > 0:
            add(
                RejectionCode.MULTIPLE_PEOPLE,
                {
                    "detected_person_count": 1 + secondary_pose_count,
                    "required_person_count": 1,
                    "detector": "MediaPipe tiled pose scan",
                },
            )

        if pose.pose_count == 1 and pose.full_body_visible:
            add(
                RejectionCode.FULL_BODY_VISIBLE,
                {
                    "knees_visible": bool(pose.knees_visible),
                    "ankles_visible": bool(pose.ankles_visible),
                },
            )

        if quality.blurry:
            add(
                RejectionCode.IMAGE_TOO_BLURRY,
                {
                    "blur_score": round(quality.blur_score, 2),
                    "minimum_blur_score": self.s.minimum_blur_score,
                },
            )

        return issues

    def validate(self, loaded: LoadedImage) -> dict:
        started = time.perf_counter()
        inference = _resize_for_inference(loaded.analysis_rgb, self.s.inference_maximum_side)

        face_candidates = self._detect_face_candidates(inference)
        pose_candidates = self._detect_pose_candidates(inference)

        if face_candidates:
            primary_index = self._primary_face_index(face_candidates)
            primary_face_candidate = face_candidates[primary_index]
            face = self._build_face_analysis(
                primary_face_candidate, face_candidates, loaded.width, loaded.height
            )
            face = self._apply_orientation(face, primary_face_candidate)
        else:
            face = FaceAnalysis()

        pose = self._pose_analysis(pose_candidates, face) if face_candidates else PoseAnalysis(
            pose_count=len(pose_candidates)
        )

        secondary_pose_count = 0
        primary_pose = None
        if face_candidates and pose_candidates and face.landmarks is not None:
            face_box = _bbox_from_points(face.landmarks)
            face_center = _bbox_center(face_box)

            def pose_distance(candidate: PoseCandidate) -> float:
                return float(np.linalg.norm(_bbox_center(candidate.bbox) - face_center))

            primary_pose = min(pose_candidates, key=pose_distance)
            primary_area = _bbox_area(primary_pose.bbox)

            for candidate in pose_candidates:
                if candidate is primary_pose:
                    continue

                candidate_area = _bbox_area(candidate.bbox)

                # Filter out minor candidates (e.g. background noise/reflections < 15% of primary area)
                if primary_area > 0 and (candidate_area / primary_area) < 0.15:
                    continue

                if not _boxes_same_person(
                    candidate.bbox,
                    primary_pose.bbox,
                    self.s.person_dedup_iou_threshold,
                    self.s.person_dedup_center_threshold,
                ):
                    secondary_pose_count += 1
        elif pose_candidates and not face_candidates:
            secondary_pose_count = max(0, len(pose_candidates) - 1)

        quality = self._quality(loaded, face)
        issues = self._issues(
            face,
            pose,
            quality,
            face_scan_count=len(face_candidates),
            secondary_pose_count=secondary_pose_count,
        )

        elapsed = round((time.perf_counter() - started) * 1000, 2)
        if issues:
            status = (
                ValidationStatus.ERROR
                if any(i.code == RejectionCode.INTERNAL_ERROR for i in issues)
                else ValidationStatus.REJECTED
            )
            raise PipelineRejection(
                issues[0].code, issues[0].details, issues, status
            )

        return {
            "policy_version": "portrait-policy-v1.1.0-balanced-single-person",
            "validation_status": ValidationStatus.ACCEPTED.value,
            "validation_ms": elapsed,
            "detection": {
                "unique_face_count": len(face_candidates),
                "unique_pose_candidates": len(pose_candidates),
                "secondary_pose_count": secondary_pose_count,
                "face_detector": "MediaPipe full + tiled scan",
                "person_detector": "MediaPipe pose full + tiled scan",
            },
            "face": {k: v for k, v in asdict(face).items() if k != "landmarks"},
            "pose": {k: v for k, v in asdict(pose).items() if k != "landmarks"},
            "quality": asdict(quality),
        }