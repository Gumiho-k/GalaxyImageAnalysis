import argparse
from pathlib import Path

from src.galaxy_image_analysis.config import AppConfig, AnalysisConfig, DetectionConfig, OCRConfig
from src.galaxy_image_analysis.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Galaxy image analysis pipeline')
    parser.add_argument('-i', '--input', default='_photo.png', help='Path to the source image')
    parser.add_argument('-o', '--output-dir', default='outputs', help='Directory for outputs')
    parser.add_argument('--q0', type=float, default=0.0, help='Intrinsic thickness parameter for inclination estimation')
    parser.add_argument('--debug-ocr', action='store_true', help='Save OCR debug crops')
    parser.add_argument('--threshold', type=int, default=25, help='Binary threshold for galaxy detection')
    parser.add_argument('--close-kernel', type=int, default=15, help='Morphological close kernel size')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        source_image=Path(args.input),
        output_dir=Path(args.output_dir),
        ocr=OCRConfig(debug_ocr=args.debug_ocr),
        detection=DetectionConfig(
            threshold_value=args.threshold,
            close_kernel_size=args.close_kernel,
        ),
        analysis=AnalysisConfig(q0=args.q0),
    )
    run_pipeline(config)


if __name__ == '__main__':
    main()
