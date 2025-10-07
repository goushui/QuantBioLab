"""
Otsu Thresholding Segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from PIL import Image
import os
from scipy import signal
from pathlib import Path

os.makedirs("Results", exist_ok=True)

image_paths = [
    "images/g9_p1.jpg",
    "images/g9_p2.jpg",
    "images/g9_p3.jpg"
]

results_file = open("Results/otsu_results.txt", "w")
def get_statistics(ground_truth, prediction):
    # Ensure binary format
    result_text = ""
    ground_truth = (ground_truth > 128).astype(np.uint8)
    prediction = (prediction > 128).astype(np.uint8)

    # Resize prediction to match ground truth if necessary
    labeled = measure.label(prediction)
    num_colonies = labeled.max()
    if ground_truth.shape != prediction.shape:
        prediction = cv2.resize(prediction, (ground_truth.shape[1], ground_truth.shape[0]), interpolation=cv2.INTER_NEAREST)
    diff = np.abs(ground_truth - prediction)
    total_diff = np.sum(diff)
    false_positive_rate = np.sum((prediction == 255) & (ground_truth == 0))/np.sum(ground_truth == 0) if np.sum(ground_truth == 0) > 0 else 0
    false_negative_rate = np.sum((prediction == 0) & (ground_truth == 255))/np.sum(ground_truth == 255) if np.sum(ground_truth == 255) > 0 else 0 
    # print(f"False Positives: {false_positive_rate}, False Negatives: {false_negative_rate}")
    regions = measure.regionprops(labeled)
    areas = [r.area for r in regions]
    circularities = [(4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
    # Compute false positives and false negatives
    false_positives = np.sum((prediction == 1) & (ground_truth == 0))
    false_negatives = np.sum((prediction == 0) & (ground_truth == 1))
    total_ground_truth = np.sum(ground_truth == 1)
    total_background = np.sum(ground_truth == 0)

    false_positive_rate = false_positives / total_background if total_background > 0 else 0
    false_negative_rate = false_negatives / total_ground_truth if total_ground_truth > 0 else 0

    result_text = f"\n{image_path}\n"
    result_text += f"  Colonies: {num_colonies}\n"
    result_text += f"  Avg Area: {np.mean(areas):.1f}\n"
    result_text += f"  Avg Circularity: {np.mean(circularities):.3f}\n"
    result_text += f"  Total Diff: {total_diff/(baseline_image.sum()):.2f}\n"
    result_text += f"  False Positive Rate: {false_positive_rate}\n"
    result_text += f"  False Negatives Rate: {false_negative_rate}\n"

    return result_text
for image_path in image_paths:
    # Load image
    image = np.array(Image.open(image_path))
    gray = np.max(image, axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    possible_petri_dishes = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/8,
                               param1=100, param2=60, minRadius=gray.shape[0]//5, maxRadius=gray.shape[0]//2)
    most_centric_circle = None # the circle that is closest to the center of the image is the most likely petri dish
    min_dist_to_center = float('inf')
    center = (gray.shape[1]//2, gray.shape[0]//2)
    if possible_petri_dishes is not None:
        possible_petri_dishes = np.uint32(np.around(possible_petri_dishes))
        for possible_petri_dish in possible_petri_dishes[0, :]:
            x, y, r = possible_petri_dish
            dist_from_center = (x-center[0])**2 + (y-center[1])**2
            if dist_from_center < min_dist_to_center:
                min_dist_to_center = dist_from_center
                most_centric_circle = (x, y, r)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (most_centric_circle[0], most_centric_circle[1]), most_centric_circle[2], 255, thickness=-1)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        print(f"Warning: No circle detected in {image_path}, using full image.")

    # Blur
    normalized = gray
    conv = np.ones((9, 9))
    averages = signal.convolve2d(normalized, conv, mode='same') / 81
    normalized = normalized - averages
    max_elem = np.max(normalized)
    min_elem = np.min(normalized)

    # Blur
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0).astype(np.uint8)

    # Otsu threshold
    threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=50)
    binary = morphology.remove_small_holes(binary, area_threshold=100)
    # baseline = "ManualFiji Segmentation/" + Path(image_path).stem + "_fiji.tif"
    # baseline_image = np.array(Image.open(baseline))
    baseline = f"Ground_Truth_Masks/{Path(image_path).stem}_mask.png"
    baseline_image = np.array(Image.open(baseline))
    gt = (baseline_image > 128).astype(np.uint8) * 255
    assert(baseline_image.shape == binary.shape)
    diff = np.abs(baseline_image - binary)
    total_diff = np.sum(diff)
    # Count colonies
    labeled = measure.label(binary)
    num_colonies = labeled.max()

    # Calculate properties
    regions = measure.regionprops(labeled)
    areas = [r.area for r in regions]
    perimeters = [r.perimeter for r in regions]
    circularities = [(4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0 for r in regions]
    # Print and save results
    # create a binary mask from regions
    prediction = np.zeros_like(gray)
    for r in regions:
        # only keep regions with circularity > 0.4 and area < 5000
        if r.area < 5000 and ((4 * np.pi * r.area) / (r.perimeter ** 2) if r.perimeter > 0 else 0) > 0.2:
            prediction[r.coords[:, 0], r.coords[:, 1]] = 255

    # result_text = f"\n{image_path}\n"
    # result_text += f"  Threshold: {threshold:.1f}\n"
    # result_text += f"  Colonies: {num_colonies}\n"
    # result_text += f"  Avg Area: {np.mean(areas):.1f}\n"
    # result_text += f"  Avg Circularity: {np.mean(circularities):.3f}\n"
    # result_text += f"  Total Diff: {total_diff:.2f}\n"
    result_text = get_statistics(baseline_image, prediction)
    print(result_text, end='')
    results_file.write(result_text)

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(blurred, cmap='gray')
    axes[1].set_title('Blurred + Normalized')
    axes[1].axis('off')

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title(f'Binary (T={threshold:.1f})')
    axes[2].axis('off')

    axes[3].imshow(labeled, cmap='nipy_spectral')
    axes[3].set_title(f'Colonies (n={num_colonies})')
    axes[3].axis('off')

    plt.suptitle(f'Otsu: {image_path}')
    plt.tight_layout()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    plt.savefig(f"Results/otsu_{image_name}.png", dpi=150, bbox_inches='tight')
    plt.close()

results_file.close()