# ==============================================================================
# --- CPU MULTI-CORE ACCELERATION SETUP ---
# This script is optimized to use all available CPU cores for the heavy
# computation in the model fitting step. It uses Python's `multiprocessing`
# library to parallelize the search for the best-fit angles.
#
# The `tqdm` library is used for a clean, user-friendly progress bar.
# Install it with: pip install tqdm
# ==============================================================================
import cv2
import numpy as np
import os
import pytesseract
import re
from PIL import Image
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# Since we are not using a GPU, `xp` will always be NumPy.
xp = np

# ==============================================================================
# --- STEP 1: GALAXY SEPARATION & OCR CODE (UNCHANGED) ---
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
# --- STEP 2: GALAXY INCLINATION ANALYSIS CODE (MULTI-CORE CPU) ---
# ==============================================================================

# --- Analysis Constants ---
CYLINDER_RADIUS = 100 
CYLINDER_HEIGHT = CYLINDER_RADIUS / 10
SYNTH_IMAGE_SIZE = 512
W_GEOMETRIC = 1.5 # Weight for the geometric part of the cost
W_LUMINOSITY = 1.0  # Weight for the luminosity part of the cost

def fit_ellipse_to_image(img, is_synth=False):
    """
    Fits an ellipse using Otsu's method and returns the contour.
    """
    if is_synth:
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    FALLBACK: Fits an ellipse to dark dust lanes by inverting the image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)
    threshold_value = np.percentile(inverted_gray, 98)
    _, thresh = cv2.threshold(inverted_gray, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: raise ValueError("No dust lane contours found.")
    all_points = np.vstack(contours)
    if len(all_points) < 5: raise ValueError("Not enough points in dust lanes to fit an ellipse.")
    ellipse = cv2.fitEllipse(all_points)
    return ellipse, all_points

def get_ellipse_properties(ellipse):
    """Extracts properties from an ellipse object."""
    (cx, cy), (d1, d2), angle = ellipse
    major_axis = max(d1, d2)
    minor_axis = min(d1, d2)
    axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
    return (cx, cy), major_axis, minor_axis, axis_ratio, angle

def create_synthetic_galaxy_image_3d(inclination, pa, radius=CYLINDER_RADIUS, height=CYLINDER_HEIGHT, size=SYNTH_IMAGE_SIZE, n_points=100000):
    """
    Creates a 3D cylinder model with a smooth luminosity profile.
    """
    rand_radius = xp.sqrt(xp.random.uniform(0, radius**2, n_points))
    rand_angle = xp.random.uniform(0, 2 * xp.pi, n_points)
    x = rand_radius * xp.cos(rand_angle)
    y = rand_radius * xp.sin(rand_angle)
    z = xp.random.uniform(-height / 2, height / 2, n_points)
    points = xp.vstack((x, y, z))
    luminosity = xp.exp(-rand_radius / (radius * 0.3))

    i_rad = xp.radians(inclination)
    pa_rad = xp.radians(pa)
    Rx = xp.array([[1, 0, 0], [0, xp.cos(i_rad), -xp.sin(i_rad)], [0, xp.sin(i_rad), xp.cos(i_rad)]])
    Rz = xp.array([[xp.cos(pa_rad), -xp.sin(pa_rad), 0], [xp.sin(pa_rad), xp.cos(pa_rad), 0], [0, 0, 1]])
    rotated_points = Rz @ Rx @ points
    proj_x, proj_y = rotated_points[0, :], rotated_points[1, :]
    
    hist_range = [[-radius, radius], [-radius, radius]]
    image, _, _ = xp.histogram2d(proj_x, proj_y, bins=size, range=hist_range, weights=luminosity)
    
    blurred_cpu = cv2.GaussianBlur(image, (5, 5), 0)
    
    if np.max(blurred_cpu) > 0:
        blurred_cpu = (blurred_cpu / np.max(blurred_cpu))
        
    return blurred_cpu

# --- Worker functions for multiprocessing ---
def worker_3d_hybrid(angles, photo_gray_norm, target_axis_ratio, target_angle):
    """Calculates cost for a single (i, pa) pair for the 3D model."""
    i, pa = angles
    
    synth_image = create_synthetic_galaxy_image_3d(i, pa)
    
    synth_image_norm = synth_image * (np.sum(photo_gray_norm) / np.sum(synth_image)) if np.sum(synth_image) > 0 else synth_image
    cost_luminosity = np.mean((photo_gray_norm - synth_image_norm)**2)
    
    synth_image_u8 = (synth_image * 255).astype(np.uint8)

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
    return cost

def worker_2d_fallback(angles, target_axis_ratio, target_angle):
    """Calculates cost for a single (i, pa) pair for the 2D model."""
    i, pa = angles
    model_axis_ratio = np.cos(np.radians(i))
    error_axis_ratio = ((model_axis_ratio - target_axis_ratio) / target_axis_ratio) ** 2 if target_axis_ratio > 0 else 0
    diff_angle = min(abs(pa - target_angle), 180 - abs(pa - target_angle))
    error_angle = (diff_angle / 90) ** 2
    cost = error_axis_ratio + error_angle
    return cost

def estimate_angles_3D_hybrid_fitting(photo_gray_norm, target_axis_ratio, target_angle):
    """
    PRIMARY METHOD: Uses multiprocessing to find the best fit.
    """
    i_range = np.arange(0, 90.5, 1)
    pa_range = np.arange(0, 180.5, 1)
    angle_pairs = [(i, pa) for i in i_range for pa in pa_range]

    print(f"-> Searching for best fit with Hybrid 3D model across {len(angle_pairs)} combinations...")
    
    task = partial(worker_3d_hybrid, photo_gray_norm=photo_gray_norm, target_axis_ratio=target_axis_ratio, target_angle=target_angle)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(task, angle_pairs), total=len(angle_pairs), desc="3D Fitting"))

    min_cost_index = np.argmin(results)
    best_i, best_pa = angle_pairs[min_cost_index]
    
    return best_i, best_pa

def estimate_angles_2D_fallback(photo_gray_norm, target_axis_ratio, target_angle):
    """
    FALLBACK METHOD: Uses multiprocessing for the 2D ellipse fitting.
    """
    i_range = np.arange(0, 90.5, 0.5)
    pa_range = np.arange(0, 180.5, 0.5)
    angle_pairs = [(i, pa) for i in i_range for pa in pa_range]
    
    print(f"-> Searching for best fit with 2D model (Fallback) across {len(angle_pairs)} combinations...")

    task = partial(worker_2d_fallback, target_axis_ratio=target_axis_ratio, target_angle=target_angle)

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(task, angle_pairs), total=len(angle_pairs), desc="2D Fitting"))

    min_cost_index = np.argmin(results)
    best_i, best_pa = angle_pairs[min_cost_index]

    return best_i, best_pa

def create_derotated_view(image, inclination, pa):
    """
    Applies an inverse affine transformation to simulate a face-on view.
    """
    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2
    i_rad = np.radians(inclination)
    pa_rad = np.radians(pa)
    cos_i = np.cos(i_rad)
    scale_factor = 1 / cos_i if not np.isclose(cos_i, 0) else 1 / np.cos(np.radians(89.9))
    c, s = np.cos(pa_rad), np.sin(pa_rad)
    R_pa = np.array([[c, -s], [s, c]])
    R_minus_pa = np.array([[c, s], [-s, c]])
    S = np.array([[1, 0], [0, scale_factor]])
    M = R_pa @ S @ R_minus_pa
    center_vec = np.array([center_x, center_y])
    translation = center_vec - M @ center_vec
    affine_matrix = np.hstack((M, translation.reshape(2, 1)))
    derotated_img_bgr = cv2.warpAffine(image, affine_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return cv2.cvtColor(derotated_img_bgr, cv2.COLOR_BGR2RGB)

def analyze_single_galaxy(image_path, output_dir):
    """
    Main analysis function for a single galaxy image.
    """
    try:
        print(f"\nAnalyzing {os.path.basename(image_path)}...")
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None: raise ValueError(f"Failed to load image: {image_path}")

        try:
            photo_ellipse, photo_contour = fit_ellipse_to_image(img)
        except ValueError:
            print("-> Standard ellipse fit failed, attempting to fit to dust lanes...")
            photo_ellipse, photo_contour = fit_ellipse_to_dust_lanes(img)

        _, _, _, target_axis_ratio, target_angle = get_ellipse_properties(photo_ellipse)
        
        img_h, img_w = img.shape[:2]
        x, y, w, h = cv2.boundingRect(photo_contour)
        
        edge_tolerance = 5
        is_partial_view = (x <= edge_tolerance or 
                           y <= edge_tolerance or 
                           (x + w) >= (img_w - edge_tolerance) or 
                           (y + h) >= (img_h - edge_tolerance))
        
        photo_gray_cpu = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (SYNTH_IMAGE_SIZE, SYNTH_IMAGE_SIZE))
        photo_gray_norm = photo_gray_cpu.astype(np.float32) / 255.0

        plot_title_suffix = ""
        start_time = time.time()
        if is_partial_view:
            print("-> WARNING: Partial view detected. Using 2D fallback for stability.")
            est_i, est_pa = estimate_angles_2D_fallback(photo_gray_norm, target_axis_ratio, target_angle)
            plot_title_suffix = " (2D Fallback)"
        else:
            try:
                print("-> Attempting primary 3D Hybrid model fitting...")
                est_i, est_pa = estimate_angles_3D_hybrid_fitting(photo_gray_norm, target_axis_ratio, target_angle)
                plot_title_suffix = " (3D Hybrid Fit)"
            except Exception as e:
                print(f"-> Primary fitting failed ({e}). Using 2D ellipse fitting as a final attempt.")
                est_i, est_pa = estimate_angles_2D_fallback(photo_gray_norm, target_axis_ratio, target_angle)
                plot_title_suffix = " (2D Fallback)"

        if est_i is None:
            raise ValueError("Could not determine best-fit angles.")
        
        print(f"-> Analysis took {time.time() - start_time:.2f} seconds.")
        print(f"-> Best Fit: Inclination={est_i:.1f}°, PA={est_pa:.1f}°")
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f"Analysis for {os.path.basename(image_path)}{plot_title_suffix}", fontsize=16)

        # Plot 1: Original image with fitted ellipse
        img_with_ellipse = img.copy()
        cv2.ellipse(img_with_ellipse, photo_ellipse, (0, 255, 0), 2)
        axes[0, 0].imshow(cv2.cvtColor(img_with_ellipse, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1. Original Photo w/ Fitted Ellipse")
        axes[0, 0].axis('off')
        
        # Plot 2: Best-fit synthetic model
        best_synth_image = create_synthetic_galaxy_image_3d(est_i, est_pa)
        axes[0, 1].imshow(best_synth_image, cmap='gray')
        axes[0, 1].set_title(f"2. Best-Fit Model (i={est_i:.1f}°, PA={est_pa:.1f}°)")
        axes[0, 1].axis('off')
        
        # --- MODIFICATION: Plot 3 is now just the de-rotated view ---
        derotated_view = create_derotated_view(img, est_i, est_pa)
        axes[1, 0].imshow(derotated_view)
        axes[1, 0].set_title("3. Reconstructed Face-on View")
        axes[1, 0].axis('off')

        # Plot 4: Difference map
        best_synth_norm = best_synth_image * (np.sum(photo_gray_norm) / np.sum(best_synth_image)) if np.sum(best_synth_image) > 0 else best_synth_image
        difference = np.abs(photo_gray_norm - best_synth_norm)
        im = axes[1, 1].imshow(difference, cmap='inferno')
        axes[1, 1].set_title("4. Difference (Model vs. Target)")
        axes[1, 1].axis('off')
        fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        file_basename = os.path.splitext(os.path.basename(image_path))[0]
        plot_output_path = os.path.join(output_dir, f"Analysis_{file_basename}.png")
        plt.savefig(plot_output_path)
        plt.close(fig)
        print(f"-> Analysis complete. Plot saved to {plot_output_path}")

    except Exception as e:
        print(f"-> ERROR: Could not analyze {os.path.basename(image_path)}. Reason: {e}")
        print("   Skipping this galaxy.")

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    source_image = '_photo.png'
    output_directory = "separated_galaxies"
    
    if not os.path.exists(source_image):
        print(f"Error: The source image '{source_image}' was not found.")
    else:
        # Step 1: Separate galaxies (this is sequential and fast)
        saved_files, out_dir = find_and_separate_galaxies(source_image, output_directory)

        if not saved_files:
            print("\nAnalysis skipped because no galaxies were separated in Step 1.")
        else:
            # Analyze galaxies sequentially to avoid nested pools
            print("\n--- Starting Step 2: Analyzing each separated galaxy sequentially ---")
            
            # A standard for loop is used here. The parallelism is now handled
            # inside the analyze_single_galaxy -> estimate_angles functions.
            for galaxy_path in saved_files:
                analyze_single_galaxy(galaxy_path, out_dir)

            print("\n\n--- Pipeline Finished ---")
