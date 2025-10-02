import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse  # Import Ellipse from matplotlib.patches
import os
from io import BytesIO

# Path to the local image file
photo_url = 'separated_galaxies/NGC_3627.jpg'  # Local file path

# 3D ellipsoid semi-axes (fixed as in original concept)
A, B, C = 100, 75, 30

def download_image(url):
    # Check if the url is a local file path
    if os.path.exists(url):
        # Read the local image file
        img = cv2.imread(url, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image from {url}")
        return img
    else:
        # Handle as a remote URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image from URL {url}")
        return img

def fit_ellipse_to_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to isolate the object (adjust threshold as needed)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image")
    # Select the largest contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:  # Minimum points needed to fit an ellipse
        raise ValueError("Not enough points to fit an ellipse")
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    return ellipse

def get_ellipse_params(ellipse):
    # ellipse is (center, (major_axis, minor_axis), angle)
    _, (major_axis, minor_axis), angle = ellipse
    ratio = minor_axis / major_axis
    return ratio, angle

def project_ellipsoid(inclination, pa, semi_axes=(A, B, C)):
    # Convert angles to radians
    i_rad = np.radians(inclination)
    pa_rad = np.radians(pa)
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(i_rad), -np.sin(i_rad)],
                   [0, np.sin(i_rad), np.cos(i_rad)]])
    Rz = np.array([[np.cos(pa_rad), -np.sin(pa_rad), 0],
                   [np.sin(pa_rad), np.cos(pa_rad), 0],
                   [0, 0, 1]])
    # Generate points on the ellipsoid surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = semi_axes[0] * np.outer(np.cos(u), np.sin(v))
    y = semi_axes[1] * np.outer(np.sin(u), np.sin(v))
    z = semi_axes[2] * np.outer(np.ones(np.size(u)), np.cos(v))
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
    # Apply rotations and project to 2D (x-y plane)
    rotated = points @ Rx.T @ Rz.T
    return rotated[:, :2]  # Return x, y coordinates

def fit_ellipse_to_points(points):
    points = np.array(points, dtype=np.float32)
    points = points.reshape(-1, 1, 2)
    ellipse = cv2.fitEllipse(points)
    return ellipse

def estimate_angles_from_ellipse(ratio_photo, angle_photo):
    i_range = np.arange(0, 91, 5)  # Inclination from 0 to 90 degrees
    pa_range = np.arange(0, 181, 5)  # Position angle from 0 to 180 degrees
    best_i, best_pa = None, None
    min_cost = float('inf')
    for i in i_range:
        for pa in pa_range:
            proj_points = project_ellipsoid(i, pa)
            synth_ellipse = fit_ellipse_to_points(proj_points)
            ratio_synth, angle_synth = get_ellipse_params(synth_ellipse)
            # Compute angle difference considering 180-degree symmetry
            diff_angle = min(abs(angle_synth - angle_photo), 180 - abs(angle_synth - angle_photo))
            # Cost function
            cost = (ratio_synth - ratio_photo) ** 2 + (diff_angle / 90) ** 2
            if cost < min_cost:
                min_cost = cost
                best_i, best_pa = i, pa
    return best_i, best_pa

def main():
    # Load and process the photo
    img = download_image(photo_url)
    photo_ellipse = fit_ellipse_to_image(img)
    ratio_photo, angle_photo = get_ellipse_params(photo_ellipse)
    
    # Estimate inclination and position angles
    est_i, est_pa = estimate_angles_from_ellipse(ratio_photo, angle_photo)
    print(f"Estimated Inclination: {est_i}°, Position Angle: {est_pa}°")
    
    # Prepare the figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Ellipsoid Projections and Photo Analysis", fontsize=16)
    
    # Define angles for the 3x3 grid
    inclinations = [0, 45, 90]
    position_angles = [0, 90, 180]
    
    # Generate synthetic projections
    for row, i in enumerate(inclinations):
        for col, pa in enumerate(position_angles):
            ax = axes[row, col]
            points = project_ellipsoid(i, pa)
            ax.scatter(points[:, 0], points[:, 1], s=1)
            ellipse = fit_ellipse_to_points(points)
            (xc, yc), (major, minor), angle = ellipse
            # Use Ellipse from matplotlib.patches
            ell_patch = Ellipse((xc, yc), major, minor, angle=angle, edgecolor='red', fc='none')
            ax.add_patch(ell_patch)
            ax.set_title(f"i={i}°, PA={pa}°")
            ax.set_aspect('equal')
            ax.axis('off')
    
    # Display the photo with fitted ellipse
    ax_photo = axes[0, 3]
    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Draw the ellipse on a copy of the image
    img_with_ellipse = img.copy()
    cv2.ellipse(img_with_ellipse, photo_ellipse, (0, 255, 0), 2)
    img_with_ellipse_rgb = cv2.cvtColor(img_with_ellipse, cv2.COLOR_BGR2RGB)
    ax_photo.imshow(img_with_ellipse_rgb)
    ax_photo.set_title(f"Photo\nEst. i={est_i}°, PA={est_pa}°")
    ax_photo.axis('off')
    
    # Adjust layout and add text
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.figtext(0.99, 0.01, f"Estimated from Photo: i={est_i}°, PA={est_pa}°", 
                ha="right", fontsize=10)
    
    # Save and show the plot
    plt.savefig("2projections_angle.png")
    plt.show()

if __name__ == "__main__":
    main()