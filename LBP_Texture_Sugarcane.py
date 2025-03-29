#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:17:45 2025

@author: pvvkishore
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 15:05:13 2025

@author: pvvkishore
"""

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte

def process_images(input_root, output_root, radius=1, points=8):
    """
    Reads images from subfolders, applies LBP, and saves them in the same structure.
    
    Parameters:
        input_root (str): Root directory containing subfolders of images.
        output_root (str): Root directory to save processed images.
        radius (int): Radius of LBP.
        points (int): Number of circularly symmetric neighbor set points.
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
                
                # Read image in grayscale
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"Skipping {input_path}, unable to read.")
                    continue
                
                # Apply LBP
                lbp_image = local_binary_pattern(image, points, radius, method='uniform')
                lbp_image = img_as_ubyte(lbp_image / np.max(lbp_image))  # Normalize to 0-255
                
                # Save the processed image
                cv2.imwrite(output_path, lbp_image)
                print(f"Saved: {output_path}")

# Example usage
input_folder = "Sugarcane Leaf Dataset"  # Change this to your input folder path
output_folder = "Sugarcane Leaf Dataset_LBP_Texture"  # Change this to your output folder path
process_images(input_folder, output_folder)
