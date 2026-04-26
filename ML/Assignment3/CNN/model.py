import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # Block 1: 1 input channel (grayscale), 16 output channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 28x28 -> 14x14
        )
        
        # Block 2: 16 input channels, 32 output channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Reduces 14x14 -> 7x7
        )
        
        # Fully Connected Classifier
        self.fc = nn.Sequential(
            nn.Linear(in_features=32 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.25), # Dropout for regularization
            nn.Linear(in_features=128, out_features=10) # 10 clothing classes
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # Flatten the tensor for the Linear layer
        x = self.fc(x)
        return x