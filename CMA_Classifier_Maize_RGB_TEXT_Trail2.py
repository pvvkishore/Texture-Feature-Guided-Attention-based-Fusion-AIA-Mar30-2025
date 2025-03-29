#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 09:54:49 2025

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

# Cross-Modal Attention Module
class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossModalAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        query = self.query_proj(query)  # (B, HW, C)
        key = self.key_proj(key)        # (B, HW, C)
        value = self.value_proj(value)  # (B, HW, C)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)  # (B, HW, HW)
        attention_weights = self.softmax(attention_scores)  # (B, HW, HW)
        context = torch.matmul(attention_weights, value)  # (B, HW, C)

        return context, attention_weights

# ResNet50 With Attention
class ResNet50WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50WithAttention, self).__init__()
        # Load ResNet50 for both streams
        self.rgb_resnet = resnet50(pretrained=False)
        self.texture_resnet = resnet50(pretrained=False)

        # Replace the global pooling and FC layer in the texture ResNet
        self.texture_resnet.avgpool = nn.Identity()  # Remove global pooling for texture
        self.texture_resnet.fc = nn.Identity()       # Remove FC layer for texture
        self.rgb_resnet.fc = nn.Identity()           # Remove FC layer for RGB

        # Project grayscale features to match RGB channel dimensions
        self.texture_projection = nn.Conv2d(2048, 512, kernel_size=1)  # Project 2048 to 512

        # Attention module
        self.attention = CrossModalAttention(embed_dim=512)

        # Classifier
        self.fc = nn.Linear(512, num_classes)

    def forward(self, rgb_input, texture_input):
        # Extract features from both streams
        rgb_features = self.rgb_resnet(rgb_input)  # Shape: (B, 2048, H/32, W/32)
        texture_features = self.texture_resnet(texture_input)  # Shape: (B, 2048, H/32, W/32)

        # Project texture features to match RGB features' channels
        texture_features = self.texture_projection(texture_features)  # Shape: (B, 512, H/32, W/32)

        # Reshape for attention
        b, c, h, w = rgb_features.size()
        rgb_flattened = rgb_features.view(b, h * w, c)  # Shape: (B, HW, C)
        texture_flattened = texture_features.view(b, h * w, c)  # Shape: (B, HW, C)

        # Apply attention
        attended_features, attention_weights = self.attention(rgb_flattened, rgb_flattened, texture_flattened)

        # Reshape attended features back to spatial dimensions
        attended_features = attended_features.view(b, c, h, w)

        # Fuse attended features back into RGB stream
        fused_features = rgb_features + attended_features

        # Global Average Pooling and Classification
        pooled_features = F.adaptive_avg_pool2d(fused_features, (1, 1)).view(b, -1)  # Shape: (B, 512)
        output = self.fc(pooled_features)  # Shape: (B, num_classes)

        return output, attention_weights

# Dataset Preparation with Brightness and Contrast Adjustment
def prepare_data(data_dir, batch_size=32):
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_texture = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust brightness and contrast
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    rgb_dir = os.path.join(data_dir, 'RGB')
    texture_dir = os.path.join(data_dir, 'Texture')

    rgb_dataset = datasets.ImageFolder(rgb_dir, transform=transform_rgb)
    texture_dataset = datasets.ImageFolder(texture_dir, transform=transform_texture)

    rgb_loader = DataLoader(rgb_dataset, batch_size=batch_size, shuffle=True)
    texture_loader = DataLoader(texture_dataset, batch_size=batch_size, shuffle=True)

    return rgb_loader, texture_loader

# Training Loop
def train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for (rgb_inputs, labels), (texture_inputs, _) in zip(rgb_loader, texture_loader):
            rgb_inputs, texture_inputs, labels = rgb_inputs.to(device), texture_inputs.to(device), labels.to(device)

            # Forward pass
            outputs, _ = model(rgb_inputs, texture_inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Visualization of Attention Maps
def visualize_attention(model, rgb_loader, texture_loader, device='cuda'):
    model.eval()
    rgb_iter = iter(rgb_loader)
    texture_iter = iter(texture_loader)

    rgb_inputs, _ = next(rgb_iter)
    texture_inputs, _ = next(texture_iter)
    rgb_inputs, texture_inputs = rgb_inputs.to(device), texture_inputs.to(device)

    with torch.no_grad():
        _, attention_weights = model(rgb_inputs, texture_inputs)

    # Visualize attention maps
    for i in range(min(6, rgb_inputs.size(0))):
        attention_map = attention_weights[i].mean(dim=0).view(14, 14).cpu()  # Average over heads and reshape
        rgb_image = rgb_inputs[i].permute(1, 2, 0).cpu().numpy()

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        plt.imshow(rgb_image)
        plt.imshow(attention_map, alpha=0.5, cmap='jet')
        plt.title("Attention Map")
        plt.show()

# Main Execution
if __name__ == "__main__":
    data_dir = "Maize_RGB_Texture"
    num_classes = 3
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_loader, texture_loader = prepare_data(data_dir, batch_size)

    model = ResNet50WithAttention(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, rgb_loader, texture_loader, criterion, optimizer, num_epochs, device)
    visualize_attention(model, rgb_loader, texture_loader, device)
