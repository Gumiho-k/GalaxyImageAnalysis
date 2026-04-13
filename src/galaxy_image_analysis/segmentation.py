from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig, OCRConfig
from .ocr import fallback_ocr_method, format_galaxy_name, get_galaxy_name_primary


@dataclass
class GalaxyCrop:
    image: np.ndarray
    bounding_box: tuple[int, int, int, int]
    filename_stem: str
    output_path: Path | None = None


def detect_galaxy_contours(image: np.ndarray, cfg: DetectionConfig) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((cfg.close_kernel_size, cfg.close_kernel_size), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image.shape[0] * image.shape[1] * cfg.min_area_ratio
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]


def make_unique_stem(stem: str, used: dict[str, int]) -> str:
    count = used.get(stem, 0)
    used[stem] = count + 1
    if count == 0:
        return stem
    return f'{stem}_{count + 1}'


def extract_galaxy_crops(
    image: np.ndarray,
    detection_cfg: DetectionConfig,
    ocr_cfg: OCRConfig,
    debug_dir: Path | None = None,
) -> list[GalaxyCrop]:
    contours = detect_galaxy_contours(image, detection_cfg)
    results: list[GalaxyCrop] = []
    used_names: dict[str, int] = {}
    unnamed_count = 0

    for idx, contour in enumerate(contours, start=1):
        rotated_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rotated_rect)
        box = np.intp(box)
        x, y, w, h = cv2.boundingRect(box)

        pad = detection_cfg.crop_padding
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
        crop = image[y1:y2, x1:x2]

        raw_name = get_galaxy_name_primary(image, (x, y, w, h), ocr_cfg, debug_dir, idx)
        formatted = format_galaxy_name(raw_name)
        if formatted is None:
            fallback = fallback_ocr_method(crop, ocr_cfg)
            formatted = format_galaxy_name(fallback)

        if formatted is None:
            unnamed_count += 1
            stem = f'Unnamed_Galaxy_{unnamed_count}'
        else:
            stem = make_unique_stem(formatted, used_names)

        results.append(GalaxyCrop(image=crop, bounding_box=(x1, y1, x2 - x1, y2 - y1), filename_stem=stem))
    return results


def save_crops(crops: list[GalaxyCrop], output_dir: Path) -> list[GalaxyCrop]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[GalaxyCrop] = []
    for crop in crops:
        out = output_dir / f'{crop.filename_stem}.jpg'
        cv2.imwrite(str(out), crop.image)
        crop.output_path = out
        saved.append(crop)
    return saved
