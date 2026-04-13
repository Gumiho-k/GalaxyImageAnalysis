# ==============================================================================
# --- GPU ACCELERATION SETUP ---
# To use GPU acceleration on Arch Linux with an NVIDIA card, you need:
# 1. The CUDA Toolkit:
#    sudo pacman -S cuda
#
# 2. The CuPy library (matches your CUDA version, e.g., 12.x):
#    pip install cupy-cuda12x
#
# 3. A CUDA-enabled build of OpenCV. The standard package may not include this.
#    You might need to build it from source or find a suitable package in the AUR.
#
# The script will automatically detect if a GPU is available and use it.
# If not, it will fall back to the CPU (NumPy).
# ==============================================================================
import cv2
import numpy as np
# Default: CPU; will only switch to GPU if all checks pass.
USE_GPU = False          # CuPy
USE_OPENCV_CUDA = False   # OpenCV
xp = np

try:
    import cupy
    from cupy.cuda.runtime import CUDARuntimeError
    xp = cupy
    USE_GPU = True
    print("✅ CuPy detected. Using GPU acceleration.")

    try:
        cupy_ready = cupy.cuda.runtime.getDeviceCount() > 0
        opencv_ready = cv2.cuda.getCudaEnabledDeviceCount() > 0

        print(f"GPU availability - CuPy device(s): {cupy_ready}, OpenCV CUDA: {opencv_ready}")

        if opencv_ready:
            print("✅ OpenCV CUDA available.")
        else:
            print("⚠️ OpenCV CUDA not available (OK, continuing with CuPy).")
    except:
        print("⚠️ OpenCV CUDA check failed (ignored).")

except ImportError:
    xp = np
    USE_GPU = False
    print("⚠️ CuPy not found. Falling back to CPU.")

# --- RANDOM SEED FOR REPRODUCIBILITY ---
SEED = 42

np.random.seed(SEED)
if USE_GPU:
    xp.random.seed(SEED)

import os
import pytesseract

# --- Tesseract availability check ---
def ensure_tesseract():
    try:
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "Tesseract OCR is not available. Install it and ensure it's on your PATH. "
            "On Ubuntu/Debian: sudo apt-get install tesseract-ocr; "
            "on macOS: brew install tesseract; "
            "on Arch: sudo pacman -S tesseract."
        ) from e
import re
from PIL import Image

import matplotlib
import os
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time # To show progress

# ==============================================================================
# --- STEP 1: GALAXY SEPARATION & OCR CODE (IMPROVED CROPPING) ---
# ==============================================================================

# --- OCR CONFIGURATION ---
DEBUG_OCR = False

def preprocess_for_ocr(image_crop):
    """Prepares a small image crop for text recognition using a robust pipeline."""
    lower_white = np.array([170, 170, 170])
    upper_white = np.array([255, 255, 255])
    color_mask = cv2.inRange(image_crop, lower_white, upper_white)
    kernel_open = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_mask = np.zeros_like(opening)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0: continue
        aspect_ratio = h / float(w)
        area = cv2.contourArea(cnt)
        if 1.0 < aspect_ratio < 8.0 and 30 < area < 5000:
            cv2.drawContours(char_mask, [cnt], -1, 255, -1)
    kernel_close = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel_close)
    inverted = cv2.bitwise_not(closing)
    scale_factor = 3
    width = int(inverted.shape[1] * scale_factor)
    height = int(inverted.shape[0] * scale_factor)
    upscaled = cv2.resize(inverted, (width, height), interpolation=cv2.INTER_NEAREST)
    return upscaled

def get_galaxy_name_primary(full_image, bounding_box, debug_counter):
    """Finds and reads text using the primary method by checking all four corners."""
    x, y, w, h = bounding_box
    search_box_w, search_box_h = 300, 150
    locations_to_check = [
        ("top_left", x, y, x + search_box_w, y + search_box_h),
        ("top_right", x + w - search_box_w, y, x + w, y + search_box_h),
        ("bottom_left", x, y + h - search_box_h, x + search_box_w, y + h),
        ("bottom_right", x + w - search_box_w, y + h - search_box_h, x + w, y + h),
    ]
    for loc_name, x1, y1, x2, y2 in locations_to_check:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(full_image.shape[1], x2), min(full_image.shape[0], y2)
        label_crop = full_image[y1:y2, x1:x2]
        if label_crop.size == 0: continue
        processed_label_img = preprocess_for_ocr(label_crop)
        ocr_config = r'--psm 7 -l eng -c tessedit_char_whitelist="NGCIC 0123456789"'
        if DEBUG_OCR:
            debug_filename = f"debug_ocr_{debug_counter}_{loc_name}.png"
            if not os.path.exists("separated_galaxies"): os.makedirs("separated_galaxies")
            cv2.imwrite(os.path.join("separated_galaxies", debug_filename), processed_label_img)
        try:
            text = pytesseract.image_to_string(processed_label_img, config=ocr_config).strip()
            if "NGC" in text.upper() or "IC" in text.upper():
                return text
        except Exception:
            continue
    return None

def fallback_ocr_method(image_crop_cv2):
    """Processes the entire galaxy crop using the simpler, second OCR method."""
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image_crop_cv2, cv2.COLOR_BGR2RGB))
        img_gray = pil_img.convert('L')
        img_bw = img_gray.point(lambda x: 0 if x < 180 else 255, '1')
        text = pytesseract.image_to_string(img_bw, lang='eng').strip()
        return text
    except Exception as e:
        print(f"  - Fallback OCR Error: {e}")
        return None

def format_galaxy_name(raw_text):
    """Cleans and formats the raw OCR text into a standardized filename string."""
    if not raw_text: return None
    match = re.search(r'(NGC|IC)[\s_]*(\d[\d\s]*)', raw_text.upper())
    if match:
        prefix, numbers = match.group(1), re.sub(r'\s+', '', match.group(2))
        return f"{prefix}_{numbers}"
    clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', raw_text)
    return clean_text.strip().replace(' ', '_')

def find_and_separate_galaxies(image_path, output_dir="separated_galaxies"):
    """
    Uses a rotated bounding rectangle to ensure the entire galaxy is cropped.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}"); return [], ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img.shape[0] * img.shape[1] * 0.001
    galaxy_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    detected_count = len(galaxy_contours)
    print(f"System detected {detected_count} potential galaxies.")

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"\nProcessing and saving galaxies to the '{output_dir}' folder...")
    
    saved_file_paths = []
    unnamed_count = 0
    for i, contour in enumerate(galaxy_contours):
        rotated_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rotated_rect)
        box = np.intp(box)
        x, y, w, h = cv2.boundingRect(box)
        
        raw_name = get_galaxy_name_primary(img, (x, y, w, h), i + 1)
        final_galaxy_name = None
        
        padding = 15
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
        galaxy_crop = img[y1:y2, x1:x2]

        if raw_name:
            final_galaxy_name = format_galaxy_name(raw_name)
        else:
            fallback_raw_name = fallback_ocr_method(galaxy_crop)
            if fallback_raw_name and ("NGC" in fallback_raw_name.upper() or "IC" in fallback_raw_name.upper()):
                final_galaxy_name = format_galaxy_name(fallback_raw_name)
        
        if final_galaxy_name:
            filename = f"{final_galaxy_name}.jpg"
        else:
            unnamed_count += 1
            filename = f"Unnamed_Galaxy_{unnamed_count}.jpg"
            
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, galaxy_crop)
        print(f"  - Saved: {filename}")
        saved_file_paths.append(output_path)
        
    print(f"\nStep 1 Complete. {len(galaxy_contours)} galaxies have been separated.")
    return saved_file_paths, output_dir

# ==============================================================================
# --- STEP 2: GALAXY INCLINATION ANALYSIS CODE (GPU ACCELERATED) ---
# ==============================================================================

# --- Analysis Constants ---
CYLINDER_RADIUS = 100 
CYLINDER_HEIGHT = CYLINDER_RADIUS / 10
SYNTH_IMAGE_SIZE = 512
W_GEOMETRIC = 1.5 # Weight for the geometric part of the cost
W_LUMINOSITY = 1.0  # Weight for the luminosity part of the cost

def fit_ellipse_to_image(img, is_synth=False):
    """
    Fits an ellipse using Otsu's method and now returns the contour as well.
    For synthetic images, it uses a simple threshold.
    """
    if is_synth:
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if USE_OPENCV_CUDA:
            gpu_gray = cv2.cuda_GpuMat()
            gpu_gray.upload(gray)
            gauss_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0)
            gpu_blurred = gauss_filter.apply(gpu_gray)
            blurred = gpu_blurred.download()
        else:
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found after thresholding.")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        raise ValueError("Not enough points in the largest contour to fit an ellipse.")
    
    ellipse = cv2.fitEllipse(contour)
    return ellipse, contour

def fit_ellipse_to_dust_lanes(img):
    """
    NEW FALLBACK: Fits an ellipse to the dark dust lanes by inverting the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert the image so dust lanes are bright
    inverted_gray = cv2.bitwise_not(gray)
    
    # Use a percentile threshold to pick out the brightest parts (the dust lanes)
    threshold_value = np.percentile(inverted_gray, 98) # Isolate top 2% of pixels
    _, thresh = cv2.threshold(inverted_gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask
    kernel = np.ones((5,5),np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No dust lane contours found.")
    
    all_points = np.vstack(contours)
    if len(all_points) < 5: raise ValueError("Not enough points in dust lanes to fit an ellipse.")
    
    ellipse = cv2.fitEllipse(all_points)
    return ellipse, all_points


def get_ellipse_properties(ellipse):
    """Extracts center, axes, axis ratio, and angle from an ellipse object."""
    (cx, cy), (d1, d2), angle = ellipse
    major_axis = max(d1, d2)
    minor_axis = min(d1, d2)
    axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
    return (cx, cy), major_axis, minor_axis, axis_ratio, angle

def generate_base_cylinder_points(radius, height, n_points):
    rand_radius = xp.sqrt(xp.random.uniform(0, radius**2, n_points))
    rand_angle = xp.random.uniform(0, 2 * xp.pi, n_points)

    x = rand_radius * xp.cos(rand_angle)
    y = rand_radius * xp.sin(rand_angle)
    z = xp.random.uniform(-height / 2, height / 2, n_points)

    points = xp.vstack((x, y, z))
    luminosity = xp.exp(-rand_radius / (radius * 0.3))

    return points, luminosity

def create_synthetic_galaxy_image_3d(
    inclination,
    pa,
    points,
    luminosity,
    radius=CYLINDER_RADIUS,
    size=SYNTH_IMAGE_SIZE
):
    i_rad = xp.radians(inclination)
    pa_rad = xp.radians(pa)

    Rx = xp.array([
        [1, 0, 0],
        [0, xp.cos(i_rad), -xp.sin(i_rad)],
        [0, xp.sin(i_rad), xp.cos(i_rad)]
    ])
    Rz = xp.array([
        [xp.cos(pa_rad), -xp.sin(pa_rad), 0],
        [xp.sin(pa_rad), xp.cos(pa_rad), 0],
        [0, 0, 1]
    ])

    rotated_points = Rz @ Rx @ points
    proj_x, proj_y = rotated_points[0, :], rotated_points[1, :]

    hist_range = [[-radius, radius], [-radius, radius]]
    image, _, _ = xp.histogram2d(
        proj_x, proj_y,
        bins=size,
        range=hist_range,
        weights=luminosity
    )

    image_cpu = image.get() if USE_GPU else image
    blurred_cpu = cv2.GaussianBlur(image_cpu, (5, 5), 0)

    if np.max(blurred_cpu) > 0:
        blurred_cpu = blurred_cpu / np.max(blurred_cpu)

    return xp.asarray(blurred_cpu) if USE_GPU else blurred_cpu


def estimate_angles_3D_hybrid_fitting(photo_gray_norm, target_axis_ratio, target_angle, points, luminosity):
    points, luminosity = generate_base_cylinder_points(
    CYLINDER_RADIUS,
    CYLINDER_HEIGHT,
    n_points=100000
    )

    i_range, pa_range = np.arange(0, 91, 1), np.arange(0, 181, 1)
    best_i, best_pa, min_cost = None, None, float('inf')
    
    total_steps = len(i_range) * len(pa_range)
    current_step = 0
    
    print("-> Searching for best fit with Hybrid 3D model... (This may take a moment)")
    start_time = time.time()

    for i in i_range:
        for pa in pa_range:
            current_step += 1
            
            synth_image = create_synthetic_galaxy_image_3d(i, pa, points, luminosity)
            
            synth_image_norm = synth_image * (xp.sum(photo_gray_norm) / xp.sum(synth_image)) if xp.sum(synth_image) > 0 else synth_image
            cost_luminosity = xp.mean((photo_gray_norm - synth_image_norm)**2)
            
            synth_image_cpu = synth_image.get() if USE_GPU else synth_image
            synth_image_u8 = (synth_image_cpu * 255).astype(np.uint8)

            try:
                synth_ellipse, _ = fit_ellipse_to_image(synth_image_u8, is_synth=True)
                _, _, _, synth_axis_ratio, synth_angle = get_ellipse_properties(synth_ellipse)
                
                error_axis_ratio = ((synth_axis_ratio - target_axis_ratio) / target_axis_ratio) ** 2 if target_axis_ratio > 0 else 0
                diff_angle = min(abs(synth_angle - target_angle), 180 - abs(synth_angle - target_angle))
                error_angle = (diff_angle / 90) ** 2
                cost_geometric = error_axis_ratio + error_angle
            except ValueError:
                cost_geometric = 1e6

            cost = (W_GEOMETRIC * cost_geometric) + (W_LUMINOSITY * cost_luminosity)
            
            if cost < min_cost:
                min_cost = cost
                best_i, best_pa = i, pa
                
            if current_step % 20 == 0:
                percent_done = (current_step / total_steps) * 100
                print(f"\r   ...Progress: {percent_done:.1f}% ({current_step}/{total_steps})", end="")

    print(f"\r   ...Search complete in {time.time() - start_time:.2f} seconds.      ")
    return best_i, best_pa


def estimate_angles_2D_fallback(photo_gray_norm, target_axis_ratio, target_angle):
    """
    FALLBACK METHOD: The previous 2D ellipse fitting logic.
    """
    i_range, pa_range = np.arange(0, 91, 5), np.arange(0, 181, 5)
    best_i, best_pa, min_cost = None, None, float('inf')
    
    print("-> Searching for best fit with 2D model (Fallback)...")
    for i in i_range:
        for pa in pa_range:
            model_axis_ratio = np.cos(np.radians(i))
            error_axis_ratio = ((model_axis_ratio - target_axis_ratio) / target_axis_ratio) ** 2 if target_axis_ratio > 0 else 0
            diff_angle = min(abs(pa - target_angle), 180 - abs(pa - target_angle))
            error_angle = (diff_angle / 90) ** 2
            cost = error_axis_ratio + error_angle
            if cost < min_cost:
                min_cost, best_i, best_pa = cost, i, pa
    return best_i, best_pa


def analyze_single_galaxy(image_path, output_plot_path):
    """
    Main function for Step 2. Prioritizes the 3D hybrid fitting method.
    Uses fixed base cylinder points for reproducible synthetic rendering.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    try:
        print("-> Attempting to fit ellipse to dust lanes...")
        photo_ellipse, photo_contour = fit_ellipse_to_dust_lanes(img)
    except ValueError:
        print("-> Dust lane fit failed, falling back to standard brightness fit...")
        photo_ellipse, photo_contour = fit_ellipse_to_image(img)

    _, _, _, target_axis_ratio, target_angle = get_ellipse_properties(photo_ellipse)

    img_h, img_w = img.shape[:2]
    x, y, w, h = cv2.boundingRect(photo_contour)

    edge_tolerance = 5
    is_partial_view = (
        x <= edge_tolerance or
        y <= edge_tolerance or
        (x + w) >= (img_w - edge_tolerance) or
        (y + h) >= (img_h - edge_tolerance)
    )

    photo_gray_cpu = cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        (SYNTH_IMAGE_SIZE, SYNTH_IMAGE_SIZE)
    )
    photo_gray_norm_cpu = photo_gray_cpu.astype(np.float32) / 255.0
    photo_gray_norm = xp.asarray(photo_gray_norm_cpu) if USE_GPU else photo_gray_norm_cpu

    plot_title_suffix = ""

    try:
        if is_partial_view:
            print("-> WARNING: Partial view detected. Using 2D fallback for stability.")
            est_i, est_pa = estimate_angles_2D_fallback(
                photo_gray_norm, target_axis_ratio, target_angle
            )
            plot_title_suffix = " (2D Fallback)"
            points = None
            luminosity = None
        else:
            print("-> Attempting primary 3D Hybrid model fitting...")

            # Generate one fixed base point cloud for reproducibility
            points, luminosity = generate_base_cylinder_points(
                CYLINDER_RADIUS,
                CYLINDER_HEIGHT,
                n_points=100000
            )

            est_i, est_pa = estimate_angles_3D_hybrid_fitting(
                photo_gray_norm,
                target_axis_ratio,
                target_angle,
                points,
                luminosity
            )
            plot_title_suffix = " (3D Hybrid Fit)"

    except Exception as e:
        print(f"-> Primary fitting failed ({e}). Using 2D ellipse fitting as a final attempt.")
        est_i, est_pa = estimate_angles_2D_fallback(
            photo_gray_norm, target_axis_ratio, target_angle
        )
        plot_title_suffix = " (2D Fallback)"
        points = None
        luminosity = None

    if est_i is None:
        raise ValueError("Could not determine best-fit angles.")

    print(f"-> Best Fit: Inclination={est_i}°, PA={est_pa}°")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Analysis for {os.path.basename(image_path)}{plot_title_suffix}", fontsize=16)

    # 1. Original image with fitted ellipse
    ax = axes[0, 0]
    img_rgb_ellipse = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.ellipse(img_rgb_ellipse, photo_ellipse, (0, 255, 0), 2)
    ax.imshow(img_rgb_ellipse)
    ax.set_title("1. Original Photo w/ Fitted Ellipse")
    ax.axis("off")

    # 2. Best-fit synthetic model
    ax = axes[0, 1]
    if points is not None and luminosity is not None:
        best_synth_image = create_synthetic_galaxy_image_3d(
            est_i,
            est_pa,
            points,
            luminosity
        )
    else:
        # For 2D fallback plotting, still generate a stable synthetic image
        fallback_points, fallback_luminosity = generate_base_cylinder_points(
            CYLINDER_RADIUS,
            CYLINDER_HEIGHT,
            n_points=100000
        )
        best_synth_image = create_synthetic_galaxy_image_3d(
            est_i,
            est_pa,
            fallback_points,
            fallback_luminosity
        )

    best_synth_image_cpu = best_synth_image.get() if USE_GPU else best_synth_image
    ax.imshow(best_synth_image_cpu, cmap="gray")
    ax.set_title(f"2. Best-Fit Model (i={est_i}°, PA={est_pa}°)")
    ax.axis("off")

    # 3. Grayscale target
    ax = axes[1, 0]
    ax.imshow(photo_gray_norm_cpu, cmap="gray")
    ax.set_title("3. Grayscale Target")
    ax.axis("off")

    # 4. Difference image
    ax = axes[1, 1]
    best_synth_norm = (
        best_synth_image * (xp.sum(photo_gray_norm) / xp.sum(best_synth_image))
        if xp.sum(best_synth_image) > 0 else best_synth_image
    )
    difference = xp.abs(photo_gray_norm - best_synth_norm)
    difference_cpu = difference.get() if USE_GPU else difference

    im = ax.imshow(difference_cpu, cmap="inferno")
    ax.set_title("4. Difference (Model vs. Target)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_plot_path)
    plt.close(fig)

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================


if __name__ == "__main__":
    import argparse
    import sys
    import os

    parser = argparse.ArgumentParser(description="Galaxy separation and analysis pipeline")
    parser.add_argument("-i", "--source_image", default="_photo.png", help="Path to source image")
    parser.add_argument("-o", "--outdir", default="separated_galaxies_newest", help="Directory to write outputs")
    args = parser.parse_args()

    # Ensure Tesseract is available before any OCR calls
    try:
        ensure_tesseract()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    source_image = args.source_image
    output_directory = args.outdir

    if not os.path.exists(source_image):
        print(f"Error: The source image '{source_image}' was not found.")
        sys.exit(1)
    else:
        saved_files, out_dir = find_and_separate_galaxies(source_image, output_directory)

        if not saved_files:
            print("Analysis skipped because no galaxies were separated in Step 1.")
        else:
            print("--- Starting Step 2: Analyzing each separated galaxy ---")
            for galaxy_path in saved_files:
                try:
                    print(f"Analyzing {os.path.basename(galaxy_path)}...")
                    file_basename = os.path.splitext(os.path.basename(galaxy_path))[0]
                    plot_output_path = os.path.join(out_dir, f"Analysis_{file_basename}.png")

                    analyze_single_galaxy(galaxy_path, plot_output_path)
                    print(f"-> Analysis complete. Plot saved to {plot_output_path}")
                except Exception as e:
                    print(f"-> ERROR: Could not analyze {os.path.basename(galaxy_path)}. Reason: {e}")
                    print("   Skipping this galaxy.")
            print("--- Pipeline Finished ---")
