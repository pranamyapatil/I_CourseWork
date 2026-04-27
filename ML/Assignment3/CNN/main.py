from ..SVM.utils import load_mnist, plot_metrics, normalize
from model import FashionCNN
from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim



def get_hyperparameters() -> Tuple[float, int, float]:
    # get the hyperparameters
    lr = 0.01
    batch_size=64
    epochs = 5
    return lr, batch_size, epochs


def main() -> None:
    # hyperparameters
    learning_rate,batch_size, epochs = get_hyperparameters()

    # get data
    print("Loading FashionMNIST dataset...")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,transform = transforms.ToTensor())
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,transform = transforms.ToTensor())
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model and move it to the device
    model = FashionCNN().to(device)

    # -----------------------------
    # 4. Loss Function and Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metrics = []
    
    model.train() # Set model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader):
            # Move tensors to the configured device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute new gradients
            optimizer.step()      # Update weights
            
            running_loss += loss.item()
            
            # Print status every 300 batches
            if (i + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")


    def test_model():
        model.eval() # Set model to evaluation mode
        num_classes = 10
        
        # Initialize tensors to store TP, FP, FN for each class
        # Shape: (10,), all zeros, on the correct device
        tp_counts = torch.zeros(num_classes, device=device)
        fp_counts = torch.zeros(num_classes, device=device)
        fn_counts = torch.zeros(num_classes, device=device)
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1) # Get predictions
                total_samples += labels.size(0)
                
                # Loop through each class to calculate TP, FP, FN for this batch
                for c in range(num_classes):
                    # True if the prediction is class 'c'
                    pred_mask = (predicted == c)
                    # True if the actual label is class 'c'
                    label_mask = (labels == c)
                    
                    # True Positives: Predicted 'c' AND Actual 'c'
                    tp_counts[c] += (pred_mask & label_mask).sum().float()
                    
                    # False Positives: Predicted 'c' BUT Actual is NOT 'c'
                    fp_counts[c] += (pred_mask & ~label_mask).sum().float()
                    
                    # False Negatives: Actual is 'c' BUT Predicted is NOT 'c'
                    fn_counts[c] += (~pred_mask & label_mask).sum().float()
                    
        # -----------------------------
        # Calculate Final Metrics
        # -----------------------------
        epsilon = 1e-15 # Prevent division by zero
        
        # Calculate metrics for each class [web:90, web:97]
        precision_per_class = tp_counts / (tp_counts + fp_counts + epsilon)
        recall_per_class = tp_counts / (tp_counts + fn_counts + epsilon)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)
        
        # Macro averages (mean across all classes)
        macro_precision = precision_per_class.mean().item()
        macro_recall = recall_per_class.mean().item()
        macro_f1 = f1_per_class.mean().item()
        
        # Global Accuracy
        global_accuracy = (tp_counts.sum() / total_samples).item() * 100
        
        print("\n--- Test Set Results (Boolean Method) ---")
        print(f"Accuracy:  {global_accuracy:.2f}%")
        print(f"Macro Precision: {macro_precision:.4f}")
        print(f"Macro Recall:    {macro_recall:.4f}")
        print(f"Macro F1 Score:  {macro_f1:.4f}")
    # plot and save the results
    # plot_metrics(metrics)


if __name__ == '__main__':
    main()




'''

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# -----------------------------
# 1. Hyperparameters & Setup
# -----------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# 2. Data Loading & Transforms
# -----------------------------
# Convert to tensor and apply z-transform normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=FASHION_MNIST_MEAN, std=FASHION_MNIST_STD)
])

# Download and load training dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# Download and load testing dataset
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)
test_loader = DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)

# -----------------------------
# 3. Define the CNN Architecture
# -----------------------------
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

# Instantiate the model and move it to the device
model = FashionCNN().to(device)

# -----------------------------
# 4. Loss Function and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# 5. Training Loop
# -----------------------------
def train_model():
    model.train() # Set model to training mode
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad() # Clear previous gradients
            loss.backward()       # Compute new gradients
            optimizer.step()      # Update weights
            
            running_loss += loss.item()
            
            # Print status every 300 batches
            if (i + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# -----------------------------
# 6. Evaluation Loop
# -----------------------------
def test_model():
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad(): # Disable gradient calculation for testing
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy of the model on the 10,000 test images: {accuracy:.2f}%")

# -----------------------------
# 7. Execution
# -----------------------------
if __name__ == '__main__':
    print("Starting Training...")
    train_model()
    test_model()


'''