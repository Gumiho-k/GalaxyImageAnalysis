from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCRConfig:
    debug_ocr: bool = False
    search_box_w: int = 300
    search_box_h: int = 150
    white_lower: tuple[int, int, int] = (170, 170, 170)
    white_upper: tuple[int, int, int] = (255, 255, 255)
    fallback_threshold: int = 180


@dataclass(frozen=True)
class DetectionConfig:
    threshold_value: int = 25
    close_kernel_size: int = 15
    min_area_ratio: float = 0.001
    crop_padding: int = 15


@dataclass(frozen=True)
class AnalysisConfig:
    blur_kernel_size: int = 15
    use_otsu: bool = True
    q0: float = 0.0
    output_plot_dpi: int = 150


@dataclass(frozen=True)
class AppConfig:
    source_image: Path = Path('_photo.png')
    output_dir: Path = Path('outputs')
    ocr: OCRConfig = OCRConfig()
    detection: DetectionConfig = DetectionConfig()
    analysis: AnalysisConfig = AnalysisConfig()
