#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 17:42:39 2025

@author: pvvkishore
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torchvision.models import resnet50
from itertools import zip_longest
from PIL import ImageFile, Image
from pathlib import Path
import numpy as np
import cv2

# Fix for truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to remove corrupted images
def remove_corrupted_images(directory):
    for img_path in Path(directory).rglob("*.*"):
        try:
            img = Image.open(img_path)
            img.verify()  # Verify if image is corrupted
        except (Image.UnidentifiedImageError, OSError):
            print(f"Removing corrupted image: {img_path}")
            img_path.unlink()  # Delete corrupted file

# Cross-Modal Attention Module
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = self.softmax(attention_scores)
        context = torch.matmul(attention_weights, value)
        return context, attention_weights

# ResNet50 With Attention - All layers
class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithAttention, self).__init__()
        self.rgb_resnet = resnet50(pretrained=False)
        self.texture_resnet = resnet50(pretrained=False)

        # Dimensions for different ResNet blocks
        self.block_dims = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
        
        # Create projection layers for each block
        self.texture_projections = nn.ModuleDict({
            'layer1': nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=1)
            ),
            'layer2': nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=1)
            ),
            'layer3': nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 1024, kernel_size=1)
            ),
            'layer4': nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(1024, 2048, kernel_size=1)
            )
        })

        # Create attention modules for all blocks
        self.attentions = nn.ModuleDict({
            'layer1': CrossModalAttention(embed_dim=256),
            'layer2': CrossModalAttention(embed_dim=512),
            'layer3': CrossModalAttention(embed_dim=1024),
            'layer4': CrossModalAttention(embed_dim=2048)
        })
        
        self.fc = nn.Linear(2048, num_classes)
        
        # Storage for attention weights of each layer for visualization
        self.layer1_attention_weights = None
        self.layer2_attention_weights = None
        self.layer3_attention_weights = None
        self.layer4_attention_weights = None
        
        # Storage for features and attended features
        self.layer1_features = None
        self.layer2_features = None
        self.layer3_features = None
        self.layer4_features = None
        
        self.layer1_attended = None
        self.layer2_attended = None
        self.layer3_attended = None
        self.layer4_attended = None

    def _apply_attention(self, rgb_x, texture_x, layer_name):
        # Apply projection to texture features
        texture_proj = self.texture_projections[layer_name](texture_x)
        
        # Get dimensions
        b, c, h, w = rgb_x.size()
        
        # Reshape for attention calculation
        rgb_flat = rgb_x.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        texture_flat = texture_proj.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        
        # Apply attention
        attended, attention_weights = self.attentions[layer_name](rgb_flat, rgb_flat, texture_flat)
        
        # Store attention weights for visualization
        if layer_name == 'layer1':
            self.layer1_attention_weights = attention_weights
        elif layer_name == 'layer2':
            self.layer2_attention_weights = attention_weights
        elif layer_name == 'layer3':
            self.layer3_attention_weights = attention_weights
        elif layer_name == 'layer4':
            self.layer4_attention_weights = attention_weights
        
        # Reshape back to original dimensions
        attended = attended.transpose(1, 2).view(b, c, h, w)
        
        # Add residual connection
        rgb_x_attended = rgb_x + attended
        
        # Store attended features
        if layer_name == 'layer1':
            self.layer1_attended = rgb_x_attended
        elif layer_name == 'layer2':
            self.layer2_attended = rgb_x_attended
        elif layer_name == 'layer3':
            self.layer3_attended = rgb_x_attended
        elif layer_name == 'layer4':
            self.layer4_attended = rgb_x_attended
            
        return rgb_x_attended

    def _forward_impl(self, rgb_input, texture_input):
        # Initial layers
        rgb_x = self.rgb_resnet.conv1(rgb_input)
        rgb_x = self.rgb_resnet.bn1(rgb_x)
        rgb_x = self.rgb_resnet.relu(rgb_x)
        rgb_x = self.rgb_resnet.maxpool(rgb_x)

        texture_x = self.texture_resnet.conv1(texture_input)
        texture_x = self.texture_resnet.bn1(texture_x)
        texture_x = self.texture_resnet.relu(texture_x)
        texture_x = self.texture_resnet.maxpool(texture_x)

        # Layer 1 with attention
        rgb_x = self.rgb_resnet.layer1(rgb_x)
        texture_x = self.texture_resnet.layer1(texture_x)
        self.layer1_features = rgb_x  # Store for Grad-CAM
        rgb_x = self._apply_attention(rgb_x, texture_x, 'layer1')

        # Layer 2 with attention
        rgb_x = self.rgb_resnet.layer2(rgb_x)
        texture_x = self.texture_resnet.layer2(texture_x)
        self.layer2_features = rgb_x  # Store for Grad-CAM
        rgb_x = self._apply_attention(rgb_x, texture_x, 'layer2')

        # Layer 3 with attention
        rgb_x = self.rgb_resnet.layer3(rgb_x)
        texture_x = self.texture_resnet.layer3(texture_x)
        self.layer3_features = rgb_x  # Store for Grad-CAM
        rgb_x = self._apply_attention(rgb_x, texture_x, 'layer3')

        # Layer 4 with attention
        rgb_x = self.rgb_resnet.layer4(rgb_x)
        texture_x = self.texture_resnet.layer4(texture_x)
        self.layer4_features = rgb_x  # Store for Grad-CAM
        rgb_x = self._apply_attention(rgb_x, texture_x, 'layer4')

        # Final pooling and classification
        x = F.adaptive_avg_pool2d(rgb_x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, {
            'layer1': self.layer1_attended,
            'layer2': self.layer2_attended,
            'layer3': self.layer3_attended,
            'layer4': self.layer4_attended
        }

    def forward(self, rgb_input, texture_input):
        return self._forward_impl(rgb_input, texture_input)

# Function to generate Grad-CAM for a specified layer
def generate_gradcam(model, rgb_input, texture_input, target_layer, class_idx=None):
    model.eval()
    gradients = []
    activations = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activations(module, input, output):
        activations.append(output)

    hook_activation = target_layer.register_forward_hook(save_activations)
    hook_gradient = target_layer.register_backward_hook(save_gradients)

    # Forward pass
    output, _ = model(rgb_input, texture_input)
    
    # If class_idx is None, use the predicted class
    if class_idx is None:
        class_idx = output.argmax(dim=1)
    
    # One-hot encoding for the target class
    one_hot = torch.zeros_like(output)
    for i, idx in enumerate(class_idx):
        one_hot[i, idx] = 1
    
    # Backward pass
    model.zero_grad()
    output.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    hook_activation.remove()
    hook_gradient.remove()
    
    # Calculate gradients and activations
    grads = gradients[0]  # [B, C, H, W]
    acts = activations[0]  # [B, C, H, W]
    
    # Take channel-wise mean of gradients
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
    
    # Weight activations by gradients
    cam = torch.sum(weights * acts, dim=1)  # [B, H, W]
    
    # Apply ReLU and normalize
    cam = F.relu(cam)
    
    # Normalize per image
    batch_cams = []
    for i in range(cam.shape[0]):
        cam_i = cam[i]
        if torch.max(cam_i) > 0:  # Check for division by zero
            cam_i = cam_i / torch.max(cam_i)
        batch_cams.append(cam_i)
    
    batch_cams = torch.stack(batch_cams)
    return batch_cams.cpu().detach()

# Updated function to visualize attention maps for all layers
def visualize_attention_maps(model, rgb_loader, texture_loader, device='cuda', num_samples=2):
    model.eval()
    model.to(device)
    
    # Get a batch of data
    rgb_iter = iter(rgb_loader)
    texture_iter = iter(texture_loader)
    
    rgb_inputs, labels = next(rgb_iter)
    texture_inputs, _ = next(texture_iter)
    
    rgb_inputs = rgb_inputs.to(device)
    texture_inputs = texture_inputs.to(device)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
    
    # Get attention weights for each layer
    attention_weights = {
        'layer1': model.layer1_attention_weights,
        'layer2': model.layer2_attention_weights,
        'layer3': model.layer3_attention_weights,
        'layer4': model.layer4_attention_weights
    }
    
    # Get Grad-CAM for each layer
    grad_cams = {
        'layer1': generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer1),
        'layer2': generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer2),
        'layer3': generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer3),
        'layer4': generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer4)
    }
    
    # Visualize attention maps
    class_names = rgb_loader.dataset.classes
    
    # Create a figure with a row for each sample and columns for each layer
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4*num_samples))
    fig.suptitle("Multi-Layer Attention Visualization", fontsize=16)
    
    # Set column titles
    if num_samples > 1:
        axes[0, 0].set_title("Original Image")
        axes[0, 1].set_title("Layer 1 Attention")
        axes[0, 2].set_title("Layer 2 Attention")
        axes[0, 3].set_title("Layer 3 Attention")
        axes[0, 4].set_title("Layer 4 Attention")
    else:
        axes[0].set_title("Original Image")
        axes[1].set_title("Layer 1 Attention")
        axes[2].set_title("Layer 2 Attention")
        axes[3].set_title("Layer 3 Attention")
        axes[4].set_title("Layer 4 Attention")
    
    for i in range(min(num_samples, rgb_inputs.size(0))):
        # Original image
        img = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        if num_samples > 1:
            # Display original image
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"{class_names[labels[i]]}")
            axes[i, 0].axis("off")
        else:
            axes[0].imshow(img)
            axes[0].set_title(f"{class_names[labels[i]]}")
            axes[0].axis("off")
        
        # Display attention maps for each layer
        for j, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4'], 1):
            attn = attention_weights[layer_name][i].mean(dim=0).cpu().numpy()
            
            # Reshape attention to square for better visualization
            h = w = int(np.sqrt(attn.shape[0]))
            attn_map = attn.reshape(h, w)
            
            # Resize attention map to match image dimensions
            attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
            
            # Normalize for visualization
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            
            # Apply colormap for better visualization
            attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
            attn_heatmap = cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Create blended image with attention
            blended_attn = 0.7 * img + 0.3 * attn_heatmap
            blended_attn = np.clip(blended_attn, 0, 1)
            
            if num_samples > 1:
                axes[i, j].imshow(blended_attn)
                axes[i, j].set_title(f"{layer_name}")
                axes[i, j].axis("off")
            else:
                axes[j].imshow(blended_attn)
                axes[j].set_title(f"{layer_name}")
                axes[j].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Add a function to compare attention maps across layers
def compare_layer_attention_maps(model, rgb_loader, texture_loader, device='cuda', num_samples=2):
    model.eval()
    model.to(device)
    
    # Get a batch of data
    rgb_iter = iter(rgb_loader)
    texture_iter = iter(texture_loader)
    
    rgb_inputs, labels = next(rgb_iter)
    texture_inputs, _ = next(texture_iter)
    
    rgb_inputs = rgb_inputs.to(device)
    texture_inputs = texture_inputs.to(device)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
    
    # Get attention weights for each layer
    attention_weights = {
        'layer1': model.layer1_attention_weights,
        'layer2': model.layer2_attention_weights,
        'layer3': model.layer3_attention_weights,
        'layer4': model.layer4_attention_weights
    }
    
    # Visualize attention maps side by side for comparison
    class_names = rgb_loader.dataset.classes
    
    for s in range(min(num_samples, rgb_inputs.size(0))):
        # Original image
        img = rgb_inputs[s].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Create figure with 2 rows: original+heatmaps and overlays
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Comparison of Attention Maps - {class_names[labels[s]]}", fontsize=16)
        
        # Original image in first row
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Original image in second row
        axes[1, 0].imshow(img)
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis("off")
        
        # Generate attention visualizations for each layer
        for j, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4'], 1):
            attn = attention_weights[layer_name][s].mean(dim=0).cpu().numpy()
            
            # Reshape attention to square for better visualization
            h = w = int(np.sqrt(attn.shape[0]))
            attn_map = attn.reshape(h, w)
            
            # Resize attention map to match image dimensions
            attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
            
            # Normalize for visualization
            attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
            
            # Apply colormap for better visualization
            attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
            attn_heatmap = cv2.cvtColor(attn_heatmap, cv2.COLOR_BGR2RGB) / 255.0
            
            # Create blended image with attention
            blended_attn = 0.7 * img + 0.3 * attn_heatmap
            blended_attn = np.clip(blended_attn, 0, 1)
            
            # Display heatmap in first row
            axes[0, j].imshow(attn_heatmap)
            axes[0, j].set_title(f"{layer_name} Heatmap")
            axes[0, j].axis("off")
            
            # Display overlay in second row
            axes[1, j].imshow(blended_attn)
            axes[1, j].set_title(f"{layer_name} Overlay")
            axes[1, j].axis("off")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

# Dataset Preparation
def prepare_data(data_dir, batch_size=8):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_texture = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    remove_corrupted_images(os.path.join(data_dir, 'RGB'))
    remove_corrupted_images(os.path.join(data_dir, 'Texture'))

    rgb_dataset = datasets.ImageFolder(os.path.join(data_dir, 'RGB'), transform=transform_rgb)
    texture_dataset = datasets.ImageFolder(os.path.join(data_dir, 'Texture'), transform=transform_texture)

    rgb_loader = DataLoader(rgb_dataset, batch_size=batch_size, shuffle=True)
    texture_loader = DataLoader(texture_dataset, batch_size=batch_size, shuffle=True)
    return rgb_loader, texture_loader

# Training Function with memory management
def train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs=30, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        processed_batches = 0

        for (rgb_inputs, labels), (texture_inputs, _) in zip_longest(rgb_loader, texture_loader, fillvalue=None):
            if rgb_inputs is None or texture_inputs is None:
                continue
            
            rgb_inputs, texture_inputs, labels = rgb_inputs.to(device), texture_inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs, _ = model(rgb_inputs, texture_inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            processed_batches += 1
            
            # Free up memory
            del rgb_inputs, texture_inputs, labels, outputs, loss
            torch.cuda.empty_cache()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/processed_batches:.4f}")

# Main Execution
if __name__ == "__main__":
    data_dir = "Maize_RGB_Texture"
    num_classes = 3
    batch_size = 8  # Reduced batch size to save memory
    num_epochs = 30
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_loader, texture_loader = prepare_data(data_dir, batch_size)

    model = ResNet50WithAttention(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs, device)
    
    # Visualize attention maps for all layers
    visualize_attention_maps(model, rgb_loader, texture_loader, device, num_samples=2)
    
    # Compare attention maps across layers
    compare_layer_attention_maps(model, rgb_loader, texture_loader, device, num_samples=2)