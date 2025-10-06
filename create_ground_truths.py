import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

ANNOTATION_FILE = "./Ground_Truth_Masks/annotations.xml"
OUTPUT_DIR = "./Ground_Truth_Masks/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def decode_cvat_rle(rle_str, width, height):
    rle = list(map(int, rle_str.strip().split(",")))
    pixels = []
    val = 0
    for length in rle:
        pixels.extend([val] * length)
        val = 1-val
    mask = np.array(pixels, dtype=np.uint8)
    if len(mask) < width * height:
        mask = np.pad(mask, (0, width*height - len(mask)))
    mask = mask[:width * height].reshape((height, width))
    return mask

tree = ET.parse(ANNOTATION_FILE)
root = tree.getroot()

for image_tag in root.findall("image"):
    image_name = image_tag.get("name")
    width = int(image_tag.get("width"))
    height = int(image_tag.get("height"))
    mask_total = np.zeros((height, width), dtype=np.uint8)

    for mask_tag in image_tag.findall("mask"):
        rle_str = mask_tag.get("rle")
        left = int(mask_tag.get("left"))
        top = int(mask_tag.get("top"))
        w = int(mask_tag.get("width"))
        h = int(mask_tag.get("height"))
        local_mask = decode_cvat_rle(rle_str, w, h)

        mask_total[top:top+h, left:left+w] = np.maximum(
            mask_total[top:top+h, left:left+w],
            local_mask
        )
    output_path = os.path.join(OUTPUT_DIR, os.path.splitext(image_name)[0] + "_mask.png")
    cv2.imwrite(output_path, mask_total * 255)