from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .config import AnalysisConfig


@dataclass
class EllipseAnalysisResult:
    center: tuple[float, float]
    major_axis: float
    minor_axis: float
    axis_ratio_obs: float
    position_angle_deg: float
    inclination_deg: float
    stretch_factor: float
    ellipse: tuple


def fit_ellipse_to_image(img: np.ndarray, cfg: AnalysisConfig) -> tuple:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel_size, cfg.blur_kernel_size), 0)
    threshold_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU if cfg.use_otsu else cv2.THRESH_BINARY
    thresh_val = 0 if cfg.use_otsu else 127
    _, thresh = cv2.threshold(blurred, thresh_val, 255, threshold_type)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No contours found after thresholding.')
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise ValueError('Not enough points to fit an ellipse.')
    return cv2.fitEllipse(contour)


def calculate_physical_parameters(ellipse_params: tuple, q0: float = 0.0) -> EllipseAnalysisResult:
    center, (axis_1, axis_2), position_angle_deg = ellipse_params
    major_axis = max(axis_1, axis_2)
    minor_axis = min(axis_1, axis_2)
    if major_axis <= 0:
        raise ValueError('Major axis must be positive.')

    q_obs = minor_axis / major_axis
    cos2_i_numerator = max(q_obs**2 - q0**2, 0.0)
    denom = max(1 - q0**2, 1e-8)
    cos2_i = min(max(cos2_i_numerator / denom, 0.0), 1.0)
    inclination_rad = np.arccos(np.sqrt(cos2_i))
    stretch_factor = 1.0 / np.sqrt(cos2_i) if cos2_i > 1e-8 else 1.0

    return EllipseAnalysisResult(
        center=center,
        major_axis=major_axis,
        minor_axis=minor_axis,
        axis_ratio_obs=q_obs,
        position_angle_deg=position_angle_deg,
        inclination_deg=float(np.rad2deg(inclination_rad)),
        stretch_factor=float(stretch_factor),
        ellipse=ellipse_params,
    )


def deproject_galaxy(image: np.ndarray, result: EllipseAnalysisResult) -> np.ndarray:
    h, w = image.shape[:2]
    center = result.center
    angle = result.position_angle_deg
    stretch = result.stretch_factor

    m = cv2.getRotationMatrix2D((0, 0), -angle, 1)
    m[1, :] *= stretch
    m_rot_back = cv2.getRotationMatrix2D((0, 0), angle, 1)
    m = m_rot_back @ np.vstack([m, [0, 0, 1]])
    m[0, 2] += center[0] - (m[0, 0] * center[0] + m[0, 1] * center[1])
    m[1, 2] += center[1] - (m[1, 0] * center[0] + m[1, 1] * center[1])

    return cv2.warpAffine(
        image,
        m[:2],
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_analysis_plot(
    original: np.ndarray,
    ellipse_params: tuple,
    deprojected: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_with_fit = original.copy()
    cv2.ellipse(img_with_fit, ellipse_params, (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. Original')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_with_fit, cv2.COLOR_BGR2RGB))
    axes[1].set_title('2. Fitted ellipse')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(deprojected, cv2.COLOR_BGR2RGB))
    axes[2].set_title('3. Deprojected')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
