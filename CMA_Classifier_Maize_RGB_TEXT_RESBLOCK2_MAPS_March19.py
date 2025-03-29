#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:05:37 2025

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
    def __init__(self, embed_dim=2048):  # Keep it same as ResNet50 output
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

# ResNet50 With Attention - Modified to expose intermediate features
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
            'layer2': nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=1)
            ),
            'layer4': nn.Sequential(
                nn.Conv2d(2048, 1024, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(1024, 2048, kernel_size=1)
            )
        })

        # Create attention modules for different blocks
        self.attentions = nn.ModuleDict({
            'layer2': CrossModalAttention(embed_dim=512),
            'layer4': CrossModalAttention(embed_dim=2048)
        })
        
        self.fc = nn.Linear(2048, num_classes)

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

        # Layer 1
        rgb_x = self.rgb_resnet.layer1(rgb_x)
        texture_x = self.texture_resnet.layer1(texture_x)

        # Layer 2 with attention
        rgb_x = self.rgb_resnet.layer2(rgb_x)
        texture_x = self.texture_resnet.layer2(texture_x)
        
        # Store layer2 features for potential Grad-CAM
        self.layer2_features = rgb_x
        
        # Apply attention at layer2
        texture_proj = self.texture_projections['layer2'](texture_x)
        
        b2, c2, h2, w2 = rgb_x.size()
        rgb_flat2 = rgb_x.view(b2, h2 * w2, c2)
        texture_flat2 = texture_proj.view(b2, h2 * w2, c2)
        
        attended2, _ = self.attentions['layer2'](rgb_flat2, rgb_flat2, texture_flat2)
        attended2 = attended2.view(b2, c2, h2, w2)
        rgb_x = rgb_x + attended2
        
        # Save the attention-enhanced features from layer2
        self.layer2_attended = rgb_x

        # Layer 3
        rgb_x = self.rgb_resnet.layer3(rgb_x)
        texture_x = self.texture_resnet.layer3(texture_x)

        # Layer 4 with attention
        rgb_x = self.rgb_resnet.layer4(rgb_x)
        texture_x = self.texture_resnet.layer4(texture_x)
        
        texture_proj = self.texture_projections['layer4'](texture_x)
        
        b4, c4, h4, w4 = rgb_x.size()
        rgb_flat4 = rgb_x.view(b4, h4 * w4, c4)
        texture_flat4 = texture_proj.view(b4, h4 * w4, c4)
        
        attended4, _ = self.attentions['layer4'](rgb_flat4, rgb_flat4, texture_flat4)
        attended4 = attended4.view(b4, c4, h4, w4)
        rgb_x = rgb_x + attended4

        # Final pooling and classification
        x = F.adaptive_avg_pool2d(rgb_x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x, self.layer2_attended

    def forward(self, rgb_input, texture_input):
        return self._forward_impl(rgb_input, texture_input)

# Improved Grad-CAM Implementation
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

# Improved visualization function for layer2
def visualize_gradcam(model, rgb_loader, texture_loader, device='cuda', num_samples=4, alpha=0.6, layer='layer2'):
    model.to(device)
    
    # Select the target layer based on input parameter
    if layer == 'layer2':
        target_layer = model.rgb_resnet.layer2
    elif layer == 'layer4':
        target_layer = model.rgb_resnet.layer4
    else:
        raise ValueError(f"Unsupported layer: {layer}")
    
    # Get a batch of data
    rgb_iter = iter(rgb_loader)
    texture_iter = iter(texture_loader)
    
    rgb_inputs, labels = next(rgb_iter)
    texture_inputs, _ = next(texture_iter)
    
    rgb_inputs = rgb_inputs.to(device)
    texture_inputs = texture_inputs.to(device)
    
    # Generate class activation maps
    cams = generate_gradcam(model, rgb_inputs, texture_inputs, target_layer)
    
    # Get predictions
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
    
    # Visualize a subset of images
    class_names = rgb_loader.dataset.classes
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    fig.suptitle(f"Grad-CAM Visualizations from {layer}", fontsize=16)
    
    for i in range(min(num_samples, rgb_inputs.size(0))):
        # Original image
        img = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Resize CAM to match image dimensions
        cam = cams[i].numpy()
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply jet colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Create blended image (original + heatmap)
        blended = (1-alpha) * img + alpha * heatmap
        blended = np.clip(blended, 0, 1)
        
        # Just the heatmap
        heatmap_only = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_only = cv2.cvtColor(heatmap_only, cv2.COLOR_BGR2RGB) / 255.0
        
        # Display images
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {class_names[labels[i]]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(heatmap_only)
        axes[i, 1].set_title(f"Grad-CAM ({layer})")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title(f"Prediction: {class_names[preds[i].item()]}")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()
    
    return cams

# Dataset Preparation
def prepare_data(data_dir, batch_size=16):
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

# Training Function
def train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for (rgb_inputs, labels), (texture_inputs, _) in zip_longest(rgb_loader, texture_loader, fillvalue=None):
            if rgb_inputs is None or texture_inputs is None:
                continue
            
            rgb_inputs, texture_inputs, labels = rgb_inputs.to(device), texture_inputs.to(device), labels.to(device)
            outputs, _ = model(rgb_inputs, texture_inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")


# Function to compare Grad-CAM visualizations from different layers
def compare_layer_gradcams(model, rgb_loader, texture_loader, device='cuda', num_samples=3):
    model.to(device)
    
    rgb_iter = iter(rgb_loader)
    texture_iter = iter(texture_loader)
    
    rgb_inputs, labels = next(rgb_iter)
    texture_inputs, _ = next(texture_iter)
    
    rgb_inputs = rgb_inputs.to(device)
    texture_inputs = texture_inputs.to(device)
    
    # Generate predictions
    with torch.no_grad():
        outputs, _ = model(rgb_inputs, texture_inputs)
        _, preds = torch.max(outputs, 1)
    
    # Generate Grad-CAMs for both layers
    layer2_cams = generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer2)
    layer4_cams = generate_gradcam(model, rgb_inputs, texture_inputs, model.rgb_resnet.layer4)
    
    # Visualize comparisons
    class_names = rgb_loader.dataset.classes
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    fig.suptitle("Comparing Grad-CAM from Different Layers", fontsize=16)
    
    for i in range(min(num_samples, rgb_inputs.size(0))):
        # Original image
        img = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        
        # Resize CAMs to match image dimensions
        cam2 = layer2_cams[i].numpy()
        cam2_resized = cv2.resize(cam2, (img.shape[1], img.shape[0]))
        heatmap2 = cv2.applyColorMap(np.uint8(255 * cam2_resized), cv2.COLORMAP_JET)
        heatmap2 = cv2.cvtColor(heatmap2, cv2.COLOR_BGR2RGB) / 255.0
        blended2 = 0.6 * img + 0.4 * heatmap2
        blended2 = np.clip(blended2, 0, 1)
        
        cam4 = layer4_cams[i].numpy()
        cam4_resized = cv2.resize(cam4, (img.shape[1], img.shape[0]))
        heatmap4 = cv2.applyColorMap(np.uint8(255 * cam4_resized), cv2.COLORMAP_JET)
        heatmap4 = cv2.cvtColor(heatmap4, cv2.COLOR_BGR2RGB) / 255.0
        blended4 = 0.6 * img + 0.4 * heatmap4
        blended4 = np.clip(blended4, 0, 1)
        
        # Display images
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {class_names[labels[i]]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(blended2)
        axes[i, 1].set_title(f"Layer2 Grad-CAM")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(blended4)
        axes[i, 2].set_title(f"Layer4 Grad-CAM")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()


# Main Execution
if __name__ == "__main__":
    data_dir = "Maize_RGB_Texture"
    num_classes = 3
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_loader, texture_loader = prepare_data(data_dir, batch_size)

    model = ResNet50WithAttention(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs, device)
    
    # Visualize Grad-CAM from layer2
    visualize_gradcam(model, rgb_loader, texture_loader, device, num_samples=4, layer='layer2')
    
    # Compare Grad-CAMs from different layers
    compare_layer_gradcams(model, rgb_loader, texture_loader, device)