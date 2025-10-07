"""
Canny Edge Detection Segmentation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, feature
from scipy import ndimage
from PIL import Image
import xml.etree.ElementTree as ET
import os
from pathlib import Path

os.makedirs("Results", exist_ok=True)

image_paths = [
    "images/g9_p1.jpg",
    "images/g9_p2.jpg",
    "images/g9_p3.jpg"
]

# Open results text file
results_file = open("Results/canny_results.txt", "w")

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
    print(f"False Positives: {false_positive_rate}, False Negatives: {false_negative_rate}")
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
    print(image_path)
    image = np.array(Image.open(image_path))
    # restrict field of view to central circle
    # Use opencv to detect the location of the petri dish and mask out everything outside it
    gray = np.max(image, axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    possible_petri_dishes = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/8,
                               param1=100, param2=60, minRadius=gray.shape[0]//5, maxRadius=gray.shape[0]//2)
    most_centric_circle = None # the circle that is closest to the center of the image is the most likely petri dish
    min_dist_to_center = float('inf')
    center = np.int32((gray.shape[1]//2, gray.shape[0]//2))
    if possible_petri_dishes is not None:
        possible_petri_dishes = np.uint64(np.around(possible_petri_dishes))
        for possible_petri_dish in possible_petri_dishes[0, :]:
            x, y, r = possible_petri_dish
            dist_from_center = (x-center[0])**2 + (y-center[1])**2
            print(type(x), type(center[0]))
            if dist_from_center < min_dist_to_center:
                min_dist_to_center = dist_from_center
                most_centric_circle = (x, y, r)
        mask = np.zeros_like(gray)
        cv2.circle(mask, (most_centric_circle[0], most_centric_circle[1]), most_centric_circle[2], 255, thickness=-1)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    else:
        print(f"Warning: No circle detected in {image_path}, using full image.")

    # convert the cvat annotation to a binary mask named ground truth from ManualFiji Segmentation/annotations.xml
    gt_mask = np.zeros_like(gray)
    # Load the CVAT annotations (this is a placeholder, implement loading as needed)
    cvat_annotations = []  # Load your CVAT annotations here
    for annotation in cvat_annotations:
        # Draw the annotation on the ground truth mask
        cv2.drawContours(gt_mask, [annotation], -1, 255, thickness=cv2.FILLED)


    # Normalize to 0-1
    gray_normalized = gray / 255.0

    # Adaptive thresholds
    mean = np.mean(gray_normalized)
    std = np.std(gray_normalized)
    low_thresh = max(0.08, mean - 0.5 * std)
    high_thresh = min(0.78, mean + 0.5 * std)
    print(f"Adaptive thresholds: [{low_thresh:.3f}, {high_thresh:.3f}]")

    # Canny edge detection
    edges = feature.canny(gray_normalized, sigma=2, low_threshold=low_thresh+0.01, high_threshold=high_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges_uint8 = (edges.astype(np.uint8)) * 255
    closed = cv2.morphologyEx(edges_uint8, cv2.MORPH_CLOSE, kernel, iterations=2)

    # edges = edges.astype(np.uint8) * 255

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter shapes
    mask = np.zeros_like(gray)
    good_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area > 0 and perimeter > 0 and area < 5000:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity > 0.5:  # Keep circular shapes
                good_contours.append(contour)
    cv2.drawContours(mask, good_contours, -1, 255, -1)
    # binary = ndimage.binary_fill_holes(mask).astype(np.uint8) * 255
    binary = mask
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    gt_path = f"Ground_Truth_Masks/{image_name}_mask.png"
    print(gt_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        print(f"Warning: No ground truth mask found for {image_name}")
        continue
    gt = (gt > 128).astype(np.uint8) * 255
    # resize binary to the size of gt
    if binary.shape != gt.shape:
        continue
        # binary = cv2.resize(binary, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Count colonies
    
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255
    baseline_image = gt // 255
    # print(baseline_image.shape, binary.shape)
    labeled = measure.label(binary)
    num_colonies = labeled.max()
    # Print and save results
    result_text = f"\n{image_path}\n"
    result_text += get_statistics(baseline_image, binary)
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
