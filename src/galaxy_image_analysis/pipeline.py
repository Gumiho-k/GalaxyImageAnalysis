from pathlib import Path

import cv2

from .analysis import calculate_physical_parameters, deproject_galaxy, fit_ellipse_to_image, save_analysis_plot
from .config import AppConfig
from .ocr import ensure_tesseract
from .segmentation import extract_galaxy_crops, save_crops


def run_pipeline(config: AppConfig) -> None:
    ensure_tesseract()
    image = cv2.imread(str(config.source_image))
    if image is None:
        raise FileNotFoundError(f'Could not load image at {config.source_image}')

    debug_dir = config.output_dir / 'debug_ocr' if config.ocr.debug_ocr else None
    crops = extract_galaxy_crops(image, config.detection, config.ocr, debug_dir)
    saved_crops = save_crops(crops, config.output_dir / 'separated_galaxies')

    for crop in saved_crops:
        ellipse = fit_ellipse_to_image(crop.image, config.analysis)
        result = calculate_physical_parameters(ellipse, q0=config.analysis.q0)
        deprojected = deproject_galaxy(crop.image, result)

        deproj_path = config.output_dir / 'deprojected' / f'{crop.filename_stem}_deprojected.jpg'
        deproj_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(deproj_path), deprojected)

        plot_path = config.output_dir / 'analysis_plots' / f'{crop.filename_stem}_analysis.png'
        save_analysis_plot(crop.image, ellipse, deprojected, plot_path, config.analysis.output_plot_dpi)
