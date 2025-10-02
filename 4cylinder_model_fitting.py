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

# Default to CPU mode. We will only switch to GPU if all checks pass.
USE_GPU = False
xp = np

try:
    import cupy
    from cupy_backends.cuda.api.runtime import CUDARuntimeError
    
    cupy_ready = cupy.cuda.runtime.getDeviceCount() > 0
    opencv_ready = cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    if cupy_ready and opencv_ready:
        USE_GPU = True
        xp = cupy
        print("✅ CuPy and OpenCV CUDA support detected. Running with full GPU acceleration.")
    else:
        if cupy_ready and not opencv_ready:
            print("⚠️ CuPy found a GPU, but OpenCV is not compiled with CUDA support.")
        print("   Falling back to CPU mode to prevent errors.")

except (ImportError, CUDARuntimeError):
    print("⚠️ CuPy not found or no CUDA device available. Falling back to CPU (NumPy).")


import os
import pytesseract
import re
from PIL import Image
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time # To show progress

# ==============================================================================
# --- STEP 1: GALAXY SEPARATION & OCR CODE ---
# ==============================================================================

DEBUG_OCR = False

def preprocess_for_ocr(image_crop):
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
        try:
            text = pytesseract.image_to_string(processed_label_img, config=ocr_config).strip()
            if "NGC" in text.upper() or "IC" in text.upper():
                return text
        except Exception:
            continue
    return None

def fallback_ocr_method(image_crop_cv2):
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
    if not raw_text: return None
    match = re.search(r'(NGC|IC)[\s_]*(\d[\d\s]*)', raw_text.upper())
    if match:
        prefix, numbers = match.group(1), re.sub(r'\s+', '', match.group(2))
        return f"{prefix}_{numbers}"
    clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', raw_text)
    return clean_text.strip().replace(' ', '_')

def find_and_separate_galaxies(image_path, output_dir="separated_galaxies"):
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
# --- STEP 2: GALAXY INCLINATION ANALYSIS CODE ---
# ==============================================================================

SYNTH_IMAGE_SIZE = 512

def fit_ellipse_to_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No contours found after thresholding.")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5: raise ValueError("Not enough points to fit an ellipse.")
    ellipse = cv2.fitEllipse(contour)
    return ellipse, contour

def get_ellipse_properties(ellipse):
    (cx, cy), (d1, d2), angle = ellipse
    major_axis = max(d1, d2)
    minor_axis = min(d1, d2)
    axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
    return (cx, cy), major_axis, minor_axis, axis_ratio, angle

def create_synthetic_galaxy_image(inclination, pa, size=SYNTH_IMAGE_SIZE):
    synth_image = xp.zeros((size, size), dtype=xp.float32)
    radius = size / 2.5
    x_vals, y_vals = xp.linspace(-radius, radius, size), xp.linspace(-radius, radius, size)
    xv, yv = xp.meshgrid(x_vals, y_vals)
    r = xp.sqrt(xv**2 + yv**2)
    luminosity = xp.exp(-r / (radius * 0.3)); luminosity[r > radius] = 0
    i_rad, pa_rad = xp.radians(inclination), xp.radians(pa)
    T_squash = xp.array([[1, 0], [0, xp.cos(i_rad)]])
    T_rot = xp.array([[xp.cos(pa_rad), -xp.sin(pa_rad)], [xp.sin(pa_rad), xp.cos(pa_rad)]])
    transform = T_rot @ T_squash
    try: inv_transform = xp.linalg.inv(transform)
    except xp.linalg.LinAlgError: return synth_image
    center = size / 2
    out_x, out_y = xp.linspace(-center, center, size), xp.linspace(-center, center, size)
    out_xv, out_yv = xp.meshgrid(out_x, out_y)
    source_coords = xp.stack((out_xv.flatten(), out_yv.flatten()), axis=1) @ inv_transform.T
    source_r = xp.sqrt(source_coords[:, 0].reshape(size, size)**2 + source_coords[:, 1].reshape(size, size)**2)
    synth_image = xp.exp(-source_r / (radius * 0.3)); synth_image[source_r > radius] = 0
    image_cpu = synth_image.get() if USE_GPU else synth_image
    blurred_cpu = cv2.GaussianBlur(image_cpu, (5, 5), 0)
    return xp.asarray(blurred_cpu) if USE_GPU else blurred_cpu

def estimate_angles_from_photo(target_axis_ratio, target_angle):
    i_range, pa_range = np.arange(0, 91, 5), np.arange(0, 181, 5)
    best_i, best_pa, min_cost = None, None, float('inf')
    print("-> Searching for best fit with 2D model...")
    for i in i_range:
        model_axis_ratio = np.cos(np.radians(i))
        error_axis_ratio = ((model_axis_ratio - target_axis_ratio) / target_axis_ratio) ** 2 if target_axis_ratio > 0 else 0
        for pa in pa_range:
            diff_angle = min(abs(pa - target_angle), 180 - abs(pa - target_angle))
            error_angle = (diff_angle / 90) ** 2
            cost = error_axis_ratio + error_angle
            if cost < min_cost:
                min_cost, best_i, best_pa = cost, i, pa
    return best_i, best_pa

def deproject_image(image, inclination, pa):
    """
    NEW: Applies an inverse transformation to the image to create a face-on view.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    i_rad = np.radians(inclination)
    pa_rad = np.radians(pa - 90) # Adjust PA for coordinate system
    
    # Inverse rotation matrix
    R_inv = np.array([
        [np.cos(pa_rad), np.sin(pa_rad), 0],
        [-np.sin(pa_rad), np.cos(pa_rad), 0],
        [0, 0, 1]
    ])
    
    # Inverse squash matrix (stretches along y-axis)
    S_inv = np.array([
        [1, 0, 0],
        [0, 1 / np.cos(i_rad) if np.cos(i_rad) > 1e-6 else 1e6, 0],
        [0, 0, 1]
    ])

    # Center the image for rotation
    T1 = np.array([[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]])

    # Combined inverse transformation
    transform_matrix = T2 @ R_inv @ S_inv @ T1

    # Apply the perspective warp
    deprojected = cv2.warpPerspective(image, transform_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return deprojected

def analyze_single_galaxy(image_path, output_plot_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None: raise ValueError(f"Failed to load image: {image_path}")

    photo_ellipse, _ = fit_ellipse_to_image(img)
    _, _, _, target_axis_ratio, target_angle = get_ellipse_properties(photo_ellipse)
    
    est_i, est_pa = estimate_angles_from_photo(target_axis_ratio, target_angle)
    if est_i is None: raise ValueError("Could not determine best-fit angles.")
    
    print(f"-> Best Fit: Inclination={est_i}°, PA={est_pa}°")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f"Analysis for {os.path.basename(image_path)}", fontsize=16)

    # 1. Original Photo
    ax = axes[0, 0]
    img_rgb_ellipse = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.ellipse(img_rgb_ellipse, photo_ellipse, (0, 255, 0), 2)
    ax.imshow(img_rgb_ellipse); ax.set_title("1. Original Photo w/ Fitted Ellipse"); ax.axis('off')
    
    # 2. Best-Fit Synthetic Model
    ax = axes[0, 1]
    best_synth_image = create_synthetic_galaxy_image(est_i, est_pa)
    ax.imshow(best_synth_image.get() if USE_GPU else best_synth_image, cmap='gray'); ax.set_title(f"2. Best-Fit Model (i={est_i}°, PA={est_pa}°)"); ax.axis('off')
    
    # 3. Deprojected (Face-on) View of Original
    ax = axes[1, 0]
    deprojected_img = deproject_image(img, est_i, est_pa)
    ax.imshow(cv2.cvtColor(deprojected_img, cv2.COLOR_BGR2RGB)); ax.set_title("3. Deprojected (Face-on View)"); ax.axis('off')

    # 4. Difference Image
    ax = axes[1, 1]
    photo_gray_norm_cpu = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (SYNTH_IMAGE_SIZE, SYNTH_IMAGE_SIZE)).astype(np.float32) / 255.0
    photo_gray_norm = xp.asarray(photo_gray_norm_cpu) if USE_GPU else photo_gray_norm_cpu
    best_synth_norm = best_synth_image * (xp.sum(photo_gray_norm) / xp.sum(best_synth_image)) if xp.sum(best_synth_image) > 0 else best_synth_image
    difference = xp.abs(photo_gray_norm - best_synth_norm)
    im = ax.imshow(difference.get() if USE_GPU else difference, cmap='inferno'); ax.set_title("4. Difference (Model vs. Target)"); ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_plot_path)
    plt.close(fig)

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    source_image = '_photo.png'
    output_directory = "separated_galaxies"
    
    if not os.path.exists(source_image):
        print(f"Error: The source image '{source_image}' was not found.")
    else:
        saved_files, out_dir = find_and_separate_galaxies(source_image, output_directory)
        if not saved_files:
            print("\nAnalysis skipped because no galaxies were separated in Step 1.")
        else:
            print("\n--- Starting Step 2: Analyzing each separated galaxy ---")
            for galaxy_path in saved_files:
                try:
                    print(f"\nAnalyzing {os.path.basename(galaxy_path)}...")
                    file_basename = os.path.splitext(os.path.basename(galaxy_path))[0]
                    plot_output_path = os.path.join(out_dir, f"Analysis_{file_basename}.png")
                    analyze_single_galaxy(galaxy_path, plot_output_path)
                    print(f"-> Analysis complete. Plot saved to {plot_output_path}")
                except Exception as e:
                    print(f"-> ERROR: Could not analyze {os.path.basename(galaxy_path)}. Reason: {e}")
                    print("   Skipping this galaxy.")
            print("\n\n--- Pipeline Finished ---")
