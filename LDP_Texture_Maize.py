#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:09:55 2025

@author: pvvkishore
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Define folder and subfolder paths
folder = "Maize_Dataset"
subfolder = "HEATHLY_1"
image_filename = "Image_8.jpg"

# Construct full image path
image_path = os.path.join(folder, subfolder, image_filename)

# Read the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Compute LDP texture
ldp_texture = compute_ldp(image)

# Display the original and LDP texture images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("LDP Texture Image")
plt.imshow(ldp_texture, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
