import cv2
import numpy as np
import os
import pytesseract
import re
from PIL import Image

# --- CONFIGURATION ---
# Set this to True to save the small black-and-white images that are
# being fed to the primary OCR engine. This is very useful for debugging.
DEBUG_OCR = False


# ==============================================================================
# PRIMARY OCR METHOD (From your first code)
# ==============================================================================

def preprocess_for_ocr(image_crop):
    """
    Prepares a small image crop for text recognition using a final, robust pipeline
    that intelligently isolates character-like shapes.
    """
    # 1. Forgiving Color Segmentation
    lower_white = np.array([170, 170, 170])
    upper_white = np.array([255, 255, 255])
    color_mask = cv2.inRange(image_crop, lower_white, upper_white)

    # 2. Noise Removal (Morphological Opening)
    kernel_open = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)

    # 3. Intelligent Contour Filtering
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_mask = np.zeros_like(opening)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w == 0: continue
        aspect_ratio = h / float(w)
        area = cv2.contourArea(cnt)
        if 1.0 < aspect_ratio < 8.0 and 30 < area < 5000:
            cv2.drawContours(char_mask, [cnt], -1, 255, -1)

    # 4. Character Consolidation (Morphological Closing)
    kernel_close = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel_close)

    # 5. Final Preparation
    inverted = cv2.bitwise_not(closing)
    scale_factor = 3
    width = int(inverted.shape[1] * scale_factor)
    height = int(inverted.shape[0] * scale_factor)
    upscaled = cv2.resize(inverted, (width, height), interpolation=cv2.INTER_NEAREST)
    return upscaled

def get_galaxy_name_primary(full_image, bounding_box, debug_counter):
    """
    Finds and reads text using the primary method by checking all four corners.
    Returns the raw detected text or None.
    """
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
        
        # Whitelist helps the primary method focus on expected characters
        ocr_config = r'--psm 7 -l eng -c tessedit_char_whitelist="NGCIC 0123456789"'
        
        if DEBUG_OCR:
            debug_filename = f"debug_ocr_{debug_counter}_{loc_name}.png"
            if not os.path.exists("separated_galaxies"): os.makedirs("separated_galaxies")
            cv2.imwrite(os.path.join("separated_galaxies", debug_filename), processed_label_img)

        try:
            text = pytesseract.image_to_string(processed_label_img, config=ocr_config).strip()
            if "NGC" in text.upper() or "IC" in text.upper():
                return text # Return raw text on first valid find
        except pytesseract.TesseractNotFoundError:
            print("\n[ERROR] Tesseract is not installed or not in your PATH.")
            return None
        except Exception:
            continue
    return None

# ==============================================================================
# FALLBACK OCR METHOD (From your second code)
# ==============================================================================

def fallback_ocr_method(image_crop_cv2):
    """
    Processes the entire galaxy crop using the simpler, second OCR method.
    Accepts an OpenCV image, converts it to PIL, and performs OCR.
    """
    try:
        # Convert OpenCV image (BGR) to PIL image (RGB)
        pil_img = Image.fromarray(cv2.cvtColor(image_crop_cv2, cv2.COLOR_BGR2RGB))

        # Apply simple grayscale and thresholding from the second script
        img_gray = pil_img.convert('L')
        img_bw = img_gray.point(lambda x: 0 if x < 180 else 255, '1')
        
        # Run OCR without a strict character whitelist for broader detection
        text = pytesseract.image_to_string(img_bw, lang='eng').strip()
        return text

    except Exception as e:
        print(f"  - Fallback OCR Error: {e}")
        return None

# ==============================================================================
# HELPER AND MAIN LOGIC
# ==============================================================================

def format_galaxy_name(raw_text):
    """
    Cleans and formats the raw OCR text into a standardized filename string.
    """
    if not raw_text:
        return None
        
    # Standardize format to "NGC_XXXX" or "IC_XXXX"
    match = re.search(r'(NGC|IC)[\s_]*(\d[\d\s]*)', raw_text.upper())
    if match:
        prefix = match.group(1)
        numbers = re.sub(r'\s+', '', match.group(2))
        return f"{prefix}_{numbers}"
    else:
        # As a last resort, clean up whatever text was found
        clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', raw_text)
        return clean_text.strip().replace(' ', '_')

def find_and_separate_galaxies(image_path, output_dir="separated_galaxies"):
    """
    Main function to detect, name, and save individual galaxies.
    Uses the primary OCR method and switches to a fallback if it fails.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # --- Galaxy Detection ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    closed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = img.shape[0] * img.shape[1] * 0.001
    galaxy_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # --- User Interaction ---
    detected_count = len(galaxy_contours)
    print(f"System detected {detected_count} potential galaxies.")
    try:
        user_count_str = input(f"How many galaxies do you see? (Press Enter to accept {detected_count}): ")
        if user_count_str and int(user_count_str) != detected_count:
            print(f"Warning: You counted {int(user_count_str)}, but the system found {detected_count}. Proceeding with system count.")
    except (ValueError, TypeError):
        print("Invalid input. Proceeding with system count.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"\nProcessing and saving galaxies to the '{output_dir}' folder...")

    unnamed_count = 0
    for i, contour in enumerate(galaxy_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # --- OCR ATTEMPT 1: PRIMARY METHOD ---
        raw_name = get_galaxy_name_primary(img, (x, y, w, h), i + 1)
        final_galaxy_name = None
        
        # --- Padding and Cropping (do this once) ---
        padding = 15
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
        galaxy_crop = img[y1:y2, x1:x2]

        if raw_name:
            print(f"  - Primary method succeeded for galaxy #{i+1}.")
            final_galaxy_name = format_galaxy_name(raw_name)
        else:
            # --- OCR ATTEMPT 2: FALLBACK METHOD ---
            print(f"  - Primary method failed for galaxy #{i+1}. Trying fallback...")
            fallback_raw_name = fallback_ocr_method(galaxy_crop)
            if fallback_raw_name and ("NGC" in fallback_raw_name.upper() or "IC" in fallback_raw_name.upper()):
                print(f"  - Fallback method found text: '{fallback_raw_name}'")
                final_galaxy_name = format_galaxy_name(fallback_raw_name)
            else:
                print(f"  - Fallback method also failed.")

        # --- File Naming and Saving ---
        if final_galaxy_name:
            filename = f"{final_galaxy_name}.jpg"
        else:
            unnamed_count += 1
            filename = f"Unnamed_Galaxy_{unnamed_count}.jpg"
            print(f"  - Warning: Could not read name for galaxy #{i+1}. Saving as default.")

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, galaxy_crop)
        print(f"  - Saved: {filename}")
        
    print(f"\nProcess complete. {len(galaxy_contours)} galaxies have been saved.")


if __name__ == "__main__":
    # The name of the main photo you want to process
    source_image = '_photo.png' 
    
    if not os.path.exists(source_image):
        print(f"Error: The source image '{source_image}' was not found.")
        print("Please make sure the image is in the same directory as this script.")
    else:
        find_and_separate_galaxies(source_image)