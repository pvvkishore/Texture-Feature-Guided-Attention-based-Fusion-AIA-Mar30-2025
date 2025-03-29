#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 15:36:58 2025

@author: pvvkishore
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:08:24 2025

@author: pvvkishore
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#%%
# Load datasets
dataset1 = datasets.ImageFolder('Maize_Dataset', transform=data_transforms)
dataset2 = datasets.ImageFolder('Sugarcane Leaf Dataset', transform=data_transforms)

dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)
dataloader2 = DataLoader(dataset2, batch_size=8, shuffle=True)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
from PIL import Image

def validate_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image integrity
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)

# Validate Dataset-1 and Dataset-2
validate_images('Maize_Dataset')
validate_images('Sugarcane Leaf Dataset')

#%%
# Basic Residual Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

# ResNet Model
class ResNet18(nn.Module):
    def __init__(self, num_classes_dataset1, num_classes_dataset2):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific heads
        self.fc_dataset1 = nn.Linear(512, num_classes_dataset1)
        self.fc_dataset2 = nn.Linear(512, num_classes_dataset2)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, task):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if task == 'dataset1':
            return self.fc_dataset1(x)
        elif task == 'dataset2':
            return self.fc_dataset2(x)
#%%
# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
#%%

class CrossDatasetAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossDatasetAttention, self).__init__()
        self.query_fc = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_fc = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_fc = nn.Linear(feature_dim, feature_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features1, features2):
        # Compute query, key, and value
        query = self.query_fc(features1)  # Shape: [batch, feature_dim]
        key = self.key_fc(features2)      # Shape: [batch, feature_dim]
        value = self.value_fc(features2)  # Shape: [batch, feature_dim]

        # Attention map
        attention = self.softmax(torch.matmul(query, key.T))  # Shape: [batch, batch]
        attended_features = torch.matmul(attention, value)    # Shape: [batch, feature_dim]

        # Combine attended features with the original
        enhanced_features = features1 + attended_features
        return enhanced_features

class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes_dataset1, num_classes_dataset2):
        super(ResNetWithAttention, self).__init__()
        # Shared ResNet backbone
        self.shared_backbone = ResNet18(num_classes_dataset1, num_classes_dataset2)
        self.shared_backbone.fc_dataset1 = nn.Identity()  # Remove classification heads
        self.shared_backbone.fc_dataset2 = nn.Identity()

        # Cross-dataset attention
        feature_dim = 512  # Output feature dimension from ResNet
        self.cross_attention = CrossDatasetAttention(feature_dim)

        # Task-specific heads
        self.fc_dataset1 = nn.Linear(feature_dim, num_classes_dataset1)
        self.fc_dataset2 = nn.Linear(feature_dim, num_classes_dataset2)

    def forward(self, x1, x2, task):
        # Extract features from both datasets
        features1 = self.shared_backbone(x1, task='dataset1')  # Shape: [batch, 512]
        features2 = self.shared_backbone(x2, task='dataset2')  # Shape: [batch, 512]

        # Apply cross-dataset attention
        enhanced_features = self.cross_attention(features1, features2)

        # Task-specific classification
        if task == 'dataset1':
            return self.fc_dataset1(enhanced_features)
        elif task == 'dataset2':
            return self.fc_dataset2(enhanced_features)
#%%
# Define the number of classes for each dataset
num_classes_dataset1 = 3
num_classes_dataset2 = 5

# Initialize the model
model = ResNetWithAttention(num_classes_dataset1, num_classes_dataset2)

# Move the model to the appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
#%%
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss_dataset1 = 0.0
    total_loss_dataset2 = 0.0

    for (inputs1, labels1), (inputs2, labels2) in zip(dataloader1, dataloader2):
        inputs1, labels1 = inputs1.to(device), labels1.to(device)
        inputs2, labels2 = inputs2.to(device), labels2.to(device)

        optimizer.zero_grad()

        # Forward pass for Dataset-1
        outputs1 = model(inputs1, inputs2, task='dataset1')
        loss1 = criterion(outputs1, labels1)

        # Forward pass for Dataset-2
        outputs2 = model(inputs1, inputs2, task='dataset2')
        loss2 = criterion(outputs2, labels2)

        # Combine losses and backpropagate
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        total_loss_dataset1 += loss1.item()
        total_loss_dataset2 += loss2.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss Dataset-1: {total_loss_dataset1:.4f}, Loss Dataset-2: {total_loss_dataset2:.4f}")
