from model import FashionCNN
from typing import Tuple
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# --- TPU IMPORTS ---
import torch_xla
import torch_xla.core.xla_model as xm

def get_hyperparameters() -> Tuple[float, int, int]:
    # get the hyperparameters
    lr = 0.01
    batch_size = 64
    epochs = 5
    return lr, batch_size, epochs

def main() -> None:
    # hyperparameters
    learning_rate, batch_size, epochs = get_hyperparameters()

    # get data
    print("Loading FashionMNIST dataset...")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # --- TPU DEVICE ASSIGNMENT ---
    device = xm.xla_device()
    print(f"Using device: {device}")

    # Instantiate the model and move it to the TPU device
    model = FashionCNN().to(device)

    # -----------------------------
    # 4. Loss Function and Optimizer
    # -----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Training...")
    model.train() # Set model to training mode

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_dataloader):

            # --- TPU ASYNC TRANSFER ---
            with torch_xla.step():
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad() 
            loss.backward() 

            # --- TPU OPTIMIZER STEP ---
            xm.optimizer_step(optimizer)

            # Print status every 300 batches (using item() triggers TPU sync, use sparingly)
            if (i + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    print("Starting Evaluation...")
    test_model(model, test_dataloader, device)

def test_model(model, test_loader, device):
    model.eval() # Set model to evaluation mode
    num_classes = 10

    # Initialize tensors to store TP, FP, FN for each class on the TPU
    tp_counts = torch.zeros(num_classes, device=device)
    fp_counts = torch.zeros(num_classes, device=device)
    fn_counts = torch.zeros(num_classes, device=device)
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # --- TPU ASYNC TRANSFER ---
            with torch_xla.step():
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get predictions
            total_samples += labels.size(0)

            # Loop through each class to calculate TP, FP, FN for this batch
            for c in range(num_classes):
                pred_mask = (predicted == c)
                label_mask = (labels == c)

                tp_counts[c] += (pred_mask & label_mask).sum().float()
                fp_counts[c] += (pred_mask & ~label_mask).sum().float()
                fn_counts[c] += (~pred_mask & label_mask).sum().float()

    # Calculate Final Metrics
    epsilon = 1e-15 # Prevent division by zero

    precision_per_class = tp_counts / (tp_counts + fp_counts + epsilon)
    recall_per_class = tp_counts / (tp_counts + fn_counts + epsilon)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

    # Bring tensors back to CPU for final printing using .item()
    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()
    global_accuracy = (tp_counts.sum() / total_samples).item() * 100

    print("\n--- Test Set Results ---")
    print(f"Accuracy:  {global_accuracy:.2f}%")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1 Score:  {macro_f1:.4f}")

if __name__ == '__main__':
    main()
