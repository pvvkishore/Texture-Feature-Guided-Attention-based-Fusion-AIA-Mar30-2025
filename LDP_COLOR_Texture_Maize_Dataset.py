#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:46:53 2025

@author: pvvkishore
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def compute_kirsch_gradients(image):
    """Compute Kirsch gradients for 8 directions."""
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # North
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),  # Northeast
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # East
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Southeast
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # South
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Southwest
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # West
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])   # Northwest
    ]
    gradients = [cv2.filter2D(image, cv2.CV_32F, kernel) for kernel in kernels]
    return gradients

def compute_ldp(image, top_k=3):
    """Compute Local Directional Pattern (LDP) with corrected data type handling."""
    gradients = compute_kirsch_gradients(image)
    gradients = np.stack(gradients, axis=-1)
    
    # Get top-k indices
    top_k_indices = np.argsort(-gradients, axis=-1)[..., :top_k]
    
    # Create LDP pattern with a larger integer type to avoid overflow
    ldp_pattern = np.zeros_like(image, dtype=np.int32)
    for k in range(top_k):
        ldp_pattern += (1 << k) * (top_k_indices[..., k] + 1)  # Use direction as binary weight
    
    return np.clip(ldp_pattern, 0, 255).astype(np.uint8)  # Clip to uint8 range

def enhance_image(image):
    """Enhance the brightness and contrast of an image using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def process_dataset(input_folder, output_folder):
    """Process dataset to compute color LDP textures, enhance them, and save."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_folder in tqdm(os.listdir(input_folder), desc="Processing Classes"):
        class_path = os.path.join(input_folder, class_folder)
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(output_folder, class_folder)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            # Read the RGB image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Split into color channels
            b_channel, g_channel, r_channel = cv2.split(image)

            # Compute LDP for each channel
            ldp_b = enhance_image(compute_ldp(b_channel))
            ldp_g = enhance_image(compute_ldp(g_channel))
            ldp_r = enhance_image(compute_ldp(r_channel))

            # Merge LDP channels into a color texture image
            color_texture = cv2.merge((ldp_b, ldp_g, ldp_r))

            # Save the color texture image
            output_image_path = os.path.join(output_class_path, image_name)
            cv2.imwrite(output_image_path, color_texture)

# Define the dataset input and output paths
input_dataset_folder = "Maize_Dataset"
output_dataset_folder = "Maize_Dataset_Texture"
# Process the dataset
process_dataset(input_dataset_folder, output_dataset_folder)
