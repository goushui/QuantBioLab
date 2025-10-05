"""
Otsu Thresholding Segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from PIL import Image
import os

os.makedirs("Results", exist_ok=True)

image_paths = [
    "images/g9_p1.jpg",
    "images/g9_p2.jpg",
    "images/g9_p3.jpg"
]

results_file = open("Results/otsu_results.txt", "w")

for image_path in image_paths:
    # Load image
    image = np.array(Image.open(image_path))
    gray = np.max(image, axis=2).astype(np.uint8)

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold
    threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=100)

    # Count colonies
    labeled = measure.label(binary)
    num_colonies = labeled.max()

    # Calculate properties
    regions = measure.regionprops(labeled)
    areas = [r.area for r in regions]
    perimeters = [r.perimeter for r in regions]
    circularities = [(4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]

    # Print and save results
    result_text = f"\n{image_path}\n"
    result_text += f"  Threshold: {threshold:.1f}\n"
    result_text += f"  Colonies: {num_colonies}\n"
    result_text += f"  Avg Area: {np.mean(areas):.1f}\n"
    result_text += f"  Avg Circularity: {np.mean(circularities):.3f}\n"

    print(result_text, end='')
    results_file.write(result_text)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(binary, cmap='gray')
    axes[1].set_title(f'Binary (T={threshold:.1f})')
    axes[1].axis('off')

    axes[2].imshow(labeled, cmap='nipy_spectral')
    axes[2].set_title(f'Colonies (n={num_colonies})')
    axes[2].axis('off')

    plt.suptitle(f'Otsu: {image_path}')
    plt.tight_layout()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f"Results/otsu_{image_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

results_file.close()
