#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(image, distances=[1], angles=[0], properties=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']):
    """
    Extracts GLCM features from a grayscale image by computing features for local patches.
    
    Parameters:
        image (numpy array): Grayscale image.
        distances (list): List of pixel pair distance offsets.
        angles (list): List of angles in radians.
        properties (list): List of GLCM properties to compute.
    
    Returns:
        feature_image (numpy array): Image representation of combined GLCM features.
    """
    h, w = image.shape
    patch_size = 10  # Define patch size for computing GLCM locally
    feature_image = np.zeros((h, w), dtype=np.float32)
    
    # Loop over image in patches
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.size == 0:
                continue
            
            glcm = graycomatrix(patch, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            
            feature_values = []
            for prop in properties:
                value = graycoprops(glcm, prop).mean()
                feature_values.append(value)
            
            # Assign mean of extracted features to the patch
            mean_feature_value = np.mean(feature_values)
            feature_image[i:i+patch_size, j:j+patch_size] = mean_feature_value
    
    # Normalize feature image to range 0-255
    feature_image = cv2.normalize(feature_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    print("Sample pixel values from the computed GLCM image:")
    print(feature_image[:5, :5])  # Print a small portion of the image for debugging
    
    return feature_image

def process_images(input_root, output_root):
    """
    Reads images from subfolders, converts to grayscale, applies GLCM feature extraction,
    and saves them in the same structure.
    
    Parameters:
        input_root (str): Root directory containing subfolders of images.
        output_root (str): Root directory to save processed images.
    """
    
    # Ensure the output directory exists
    os.makedirs(output_root, exist_ok=True)
    
    # Traverse input directory
    for subdir, _, files in os.walk(input_root):
        relative_path = os.path.relpath(subdir, input_root)
        output_subdir = os.path.join(output_root, relative_path)
        
        # Create corresponding subfolder in output directory
        os.makedirs(output_subdir, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(output_subdir, file)
                
                # Read image in RGB and convert to grayscale
                image_rgb = cv2.imread(input_path)
                if image_rgb is None:
                    print(f"Skipping {input_path}, unable to read.")
                    continue
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
                
                # Apply GLCM feature extraction
                glcm_image = extract_glcm_features(image_gray)
                
                # Save the processed image
                cv2.imwrite(output_path, glcm_image)
                print(f"Saved: {output_path}")

# Example usage
input_folder = "Sugarcane Leaf Dataset"  # Change this to your input folder path
output_folder = "Sugarcane Leaf Dataset_GLCM"  # Change this to your output folder path
process_images(input_folder, output_folder)
