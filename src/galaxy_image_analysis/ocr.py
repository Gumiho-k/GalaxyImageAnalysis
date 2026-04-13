import re
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from .config import OCRConfig


VALID_NAME_RE = re.compile(r'^(NGC|IC)[ _]?\d+$')


def ensure_tesseract() -> None:
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            'Tesseract OCR is not available. Install it and ensure it is on PATH.'
        ) from exc


def preprocess_for_ocr(image_crop: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    lower_white = np.array(cfg.white_lower)
    upper_white = np.array(cfg.white_upper)
    color_mask = cv2.inRange(image_crop, lower_white, upper_white)

    kernel_open = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_mask = np.zeros_like(opening)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0:
            continue
        aspect_ratio = h / float(w)
        area = cv2.contourArea(cnt)
        if 1.0 < aspect_ratio < 8.0 and 30 < area < 5000:
            cv2.drawContours(char_mask, [cnt], -1, 255, -1)

    kernel_close = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel_close)
    inverted = cv2.bitwise_not(closing)
    return cv2.resize(
        inverted,
        (inverted.shape[1] * 3, inverted.shape[0] * 3),
        interpolation=cv2.INTER_NEAREST,
    )


def format_galaxy_name(raw_text: str | None) -> str | None:
    if not raw_text:
        return None

    match = re.search(r'(NGC|IC)[\s_]*(\d[\d\s]*)', raw_text.upper())
    if match:
        prefix = match.group(1)
        numbers = re.sub(r'\s+', '', match.group(2))
        candidate = f'{prefix}_{numbers}'
        return candidate if VALID_NAME_RE.match(candidate) else None
    return None


def _ocr_image(image: np.ndarray, config: str) -> str:
    return pytesseract.image_to_string(image, config=config).strip()


def get_galaxy_name_primary(
    full_image: np.ndarray,
    bounding_box: tuple[int, int, int, int],
    cfg: OCRConfig,
    debug_dir: Path | None = None,
    debug_counter: int = 0,
) -> str | None:
    x, y, w, h = bounding_box
    locations = [
        ('top_left', x, y, x + cfg.search_box_w, y + cfg.search_box_h),
        ('top_right', x + w - cfg.search_box_w, y, x + w, y + cfg.search_box_h),
        ('bottom_left', x, y + h - cfg.search_box_h, x + cfg.search_box_w, y + h),
        ('bottom_right', x + w - cfg.search_box_w, y + h - cfg.search_box_h, x + w, y + h),
    ]

    ocr_config = r'--psm 7 -l eng -c tessedit_char_whitelist="NGCIC 0123456789"'
    for loc_name, x1, y1, x2, y2 in locations:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_image.shape[1], x2), min(full_image.shape[0], y2)
        label_crop = full_image[y1:y2, x1:x2]
        if label_crop.size == 0:
            continue

        processed = preprocess_for_ocr(label_crop, cfg)
        if cfg.debug_ocr and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f'debug_ocr_{debug_counter}_{loc_name}.png'), processed)

        try:
            text = _ocr_image(processed, ocr_config)
        except Exception:
            continue
        if 'NGC' in text.upper() or 'IC' in text.upper():
            return text
    return None


def fallback_ocr_method(image_crop: np.ndarray, cfg: OCRConfig) -> str | None:
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        img_gray = pil_img.convert('L')
        img_bw = img_gray.point(lambda x: 0 if x < cfg.fallback_threshold else 255, '1')
        return pytesseract.image_to_string(img_bw, lang='eng').strip()
    except Exception:
        return None
