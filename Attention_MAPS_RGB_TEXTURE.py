#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:16:17 2025

@author: pvvkishore
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms

# Load a pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Define hooks to capture the outputs of the layers we are interested in
features = {}

def hook_fn(name):
    def hook(module, input, output):
        features[name] = output
    return hook

# Register hooks to capture features from specific convolutional layers
model.conv1.register_forward_hook(hook_fn("initial_conv"))
model.layer1[0].conv1.register_forward_hook(hook_fn("middle_conv"))
model.layer2[0].conv1.register_forward_hook(hook_fn("final_conv"))

# Load the first image and apply the necessary transformations
image1 = Image.open('healthy (56).jpeg')
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ensure consistent feature map size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformation to the first image
image1_tensor = transform(image1).unsqueeze(0)
#%%
from torchvision.transforms import ToPILImage, Resize, Compose, ToTensor
from PIL import Image, ImageEnhance

# Function to normalize the attention map to [0, 1] range
def normalize_attention_map(attention_map):
    min_val = attention_map.min()
    max_val = attention_map.max()
    return (attention_map - min_val) / (max_val - min_val)

# Convert the original image (Image 1) to a format suitable for visualization
def convert_image_for_display(image_tensor):
    image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    image_tensor = image_tensor.permute(1, 2, 0)  # Change shape to (H, W, C)
    image_tensor = image_tensor.numpy()  # Convert to numpy array
    image_tensor = np.clip(image_tensor, 0, 1)  # Ensure values are between 0 and 1
    return image_tensor

# Function to enhance image2
def enhance_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)  # Increase brightness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2)  # Increase sharpness
    return image
#%%
# Load a 1-channel (grayscale) image
#image2 = Image.open('Image_3228T.jpg').convert('L')  # 'L' mode for grayscale
# Load and enhance image2
image2_enhanced = enhance_image('healthy (56)T.jpeg')  # Apply enhancements

# Convert the image to a tensor
transform = transforms.ToTensor()
image_tensor2 = transform(image2_enhanced)

# Check the shape of the original tensor (should be 1xHxW for a single channel)
print("Original Image Shape:", image_tensor2.shape)

# Convert to 3-channel by repeating the 1st channel 3 times
#rgb_image_tensor = image_tensor.repeat(3, 1, 1)

# Check the new shape (should be 3xHxW)
#print("Converted Image Shape:", rgb_image_tensor.shape)

# Convert the tensor back to a PIL image if needed
rgb_image2 = transforms.ToPILImage()(image_tensor2)
#rgb_image2.show()  # Display the converted image
#%%
# Load the first image and apply the necessary transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ensure consistent feature map size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply transformation to both images
image1_tensor = transform(image1).unsqueeze(0)
image2_tensor = transform(rgb_image2).unsqueeze(0)

# Ensure model is in evaluation mode
model.eval()

# Forward pass through the model for the first image
with torch.no_grad():
    _ = model(image1_tensor)
    
# Forward pass through the model for the second image
with torch.no_grad():
    _ = model(image2_tensor)

# Print out the feature map shapes for both images
print("Image 1:")
print("Initial Conv Layer Features:", features["initial_conv"].shape)
print("Middle Conv Layer Features:", features["middle_conv"].shape)
print("Final Conv Layer Features:", features["final_conv"].shape)

print("Image 2:")
print("Initial Conv Layer Features (Image 2):", features["initial_conv"].shape)
print("Middle Conv Layer Features (Image 2):", features["middle_conv"].shape)
print("Final Conv Layer Features (Image 2):", features["final_conv"].shape)
#%%
import torch
import torch.nn.functional as F

# Assuming 'features' dictionary contains the feature maps for both images
# Image 1 features (Query): from the first image, 'features["initial_conv"]'
# Image 2 features (Key, Value): from the second image, 'features["initial_conv"]'

# Define a function to compute self-attention
def compute_self_attention(query_features, key_value_features):
    # Flatten the spatial dimensions (HxW) to (N, D), where N is the number of patches and D is the feature dimension
    query_features_flat = query_features.flatten(2).transpose(1, 2)  # (1, D, H*W) -> (1, H*W, D)
    key_value_features_flat = key_value_features.flatten(2).transpose(1, 2)  # (1, D, H*W) -> (1, H*W, D)

    # Compute the attention scores (QK^T / sqrt(d_k))
    d_k = query_features_flat.size(-1)  # the dimension of the key (same as the query feature dimension)
    attention_scores = torch.bmm(query_features_flat, key_value_features_flat.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply the softmax function to normalize the attention scores (along the key dimension)
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Compute the output by applying the attention weights to the value
    attention_output = torch.bmm(attention_weights, key_value_features_flat)

    # Optional: Reshape the attention output back to the original spatial dimensions
    attention_output = attention_output.transpose(1, 2).reshape(1, query_features.size(1), query_features.size(2), query_features.size(3))

    return attention_output

# Extracting feature maps for all three layers (initial, middle, final)
query_features_initial = features["initial_conv"]  # Query (Image 1)
key_value_features_initial = features["initial_conv"]  # Key and Value (Image 2)

query_features_middle = features["middle_conv"]  # Query (Image 1)
key_value_features_middle = features["middle_conv"]  # Key and Value (Image 2)

query_features_final = features["final_conv"]  # Query (Image 1)
key_value_features_final = features["final_conv"]  # Key and Value (Image 2)

# Compute self-attention for each layer
attention_output_initial = compute_self_attention(query_features_initial, key_value_features_initial)
attention_output_middle = compute_self_attention(query_features_middle, key_value_features_middle)
attention_output_final = compute_self_attention(query_features_final, key_value_features_final)

# Print the attention output shapes for each layer
print("Attention Output for Initial Layer:", attention_output_initial.shape)
print("Attention Output for Middle Layer:", attention_output_middle.shape)
print("Attention Output for Final Layer:", attention_output_final.shape)
#%%
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage, Resize

# Function to normalize the attention map to [0, 1] range
def normalize_attention_map(attention_map):
    min_val = attention_map.min()
    max_val = attention_map.max()
    return (attention_map - min_val) / (max_val - min_val)

# Convert the original image (Image 1) to a format suitable for visualization
def convert_image_for_display(image_tensor):
    image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
    image_tensor = image_tensor.permute(1, 2, 0)  # Change shape to (H, W, C)
    image_tensor = image_tensor.numpy()  # Convert to numpy array
    image_tensor = np.clip(image_tensor, 0, 1)  # Ensure values are between 0 and 1
    return image_tensor

# Resize the attention map to match the original image size (224x224)
resize = Resize((224, 224))

# Convert attention output to PIL image and normalize it
def visualize_attention_on_image(image_tensor, attention_map, layer_name):
    # Sum or average across all channels (assuming the attention map has multiple channels)
    attention_map_reduced = attention_map.sum(dim=1)  # Sum across channels (C dimension)
    
    # Normalize and resize the reduced attention map to match original image size
    attention_map_resized = resize(ToPILImage()(attention_map_reduced.squeeze(0)))
    attention_map_resized = np.array(attention_map_resized) / 255.0  # Normalize to [0, 1]

    # Normalize attention map to [0, 1] for visualization
    attention_map_normalized = normalize_attention_map(attention_map_resized)

    # Convert original image for display
    original_image = convert_image_for_display(image_tensor)

    # Overlay the attention map on the original image (color map for visualization)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(original_image)
    ax.imshow(attention_map_normalized, cmap='jet', alpha=0.6)  # Use 'jet' colormap with transparency
    ax.set_title(f'Attention Map Overlay: {layer_name}')
    ax.axis('off')  # Hide axis
    plt.show()

# Visualize attention maps for each layer
visualize_attention_on_image(image1_tensor, attention_output_initial, "Initial Conv Layer")
visualize_attention_on_image(image1_tensor, attention_output_middle, "Middle Conv Layer")
visualize_attention_on_image(image1_tensor, attention_output_final, "Final Conv Layer")
