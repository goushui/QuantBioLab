"""
Canny Edge Detection Segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, feature
from scipy import ndimage
from PIL import Image
import os

os.makedirs("Results", exist_ok=True)

image_paths = [
    "images/g9_p1.jpg",
    "images/g9_p2.jpg",
    "images/g9_p3.jpg"
]

# Open results text file
results_file = open("Results/canny_results.txt", "w")

for image_path in image_paths:
    # Load image
    image = np.array(Image.open(image_path))
    gray = np.max(image, axis=2).astype(np.uint8)

    # Normalize to 0-1
    gray_normalized = gray / 255.0

    # Adaptive thresholds
    mean = np.mean(gray_normalized)
    std = np.std(gray_normalized)
    low_thresh = max(0.08, mean - 0.5 * std)
    high_thresh = min(0.78, mean + 0.5 * std)

    # Canny edge detection
    edges = feature.canny(gray_normalized, sigma=2, low_threshold=low_thresh, high_threshold=high_thresh)
    edges = edges.astype(np.uint8) * 255

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter shapes
    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 50 and perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.5:  # Keep circular shapes
                good_contours.append(contour)

    # Draw filled contours
    binary = np.zeros_like(gray)
    cv2.drawContours(binary, good_contours, -1, 255, thickness=cv2.FILLED)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Count colonies
    labeled = measure.label(binary)
    num_colonies = labeled.max()

    # Calculate properties
    regions = measure.regionprops(labeled)
    areas = [r.area for r in regions]
    circularities = [(4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]

    # Print and save results
    result_text = f"\n{image_path}\n"
    result_text += f"  Thresholds: [{low_thresh:.3f}, {high_thresh:.3f}]\n"
    result_text += f"  Colonies: {num_colonies}\n"
    result_text += f"  Avg Area: {np.mean(areas):.1f}\n"
    result_text += f"  Avg Circularity: {np.mean(circularities):.3f}\n"

    results_file.write(result_text)
    print(result_text, end='')

    # Show results
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title(f'Edges [{low_thresh:.3f},{high_thresh:.3f}]')
    axes[1].axis('off')

    axes[2].imshow(binary, cmap='gray')
    axes[2].set_title('Binary')
    axes[2].axis('off')

    axes[3].imshow(labeled, cmap='nipy_spectral')
    axes[3].set_title(f'Colonies (n={num_colonies})')
    axes[3].axis('off')

    plt.suptitle(f'Canny: {image_path}')
    plt.tight_layout()

    # Save figure
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f"Results/canny_{image_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

results_file.close()
