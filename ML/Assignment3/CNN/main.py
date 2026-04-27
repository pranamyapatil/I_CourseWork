from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import FashionCNN


def get_hyperparameters() -> Tuple[float, int, int]:
    learning_rate = 0.001
    batch_size = 64
    epochs = 5
    return learning_rate, batch_size, epochs


def train_model(model, train_dataloader, criterion, optimizer, device, epochs, scheduler=None):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 300 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Batch [{i + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {epoch_loss:.4f}")

        if scheduler is not None:
            scheduler.step()


def test_model(model, test_dataloader, device):
    model.eval()
    num_classes = 10

    tp_counts = torch.zeros(num_classes, device=device)
    fp_counts = torch.zeros(num_classes, device=device)
    fn_counts = torch.zeros(num_classes, device=device)
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)

            for c in range(num_classes):
                pred_mask = predicted == c
                label_mask = labels == c

                tp_counts[c] += (pred_mask & label_mask).sum().float()
                fp_counts[c] += (pred_mask & ~label_mask).sum().float()
                fn_counts[c] += (~pred_mask & label_mask).sum().float()

    epsilon = 1e-15

    precision_per_class = tp_counts / (tp_counts + fp_counts + epsilon)
    recall_per_class = tp_counts / (tp_counts + fn_counts + epsilon)
    f1_per_class = 2 * (precision_per_class * recall_per_class) / (
        precision_per_class + recall_per_class + epsilon
    )

    macro_precision = precision_per_class.mean().item()
    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()
    global_accuracy = (tp_counts.sum() / total_samples).item() * 100

    print("\n--- Test Set Results ---")
    print(f"Accuracy: {global_accuracy:.2f}%")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    return global_accuracy, macro_precision, macro_recall, macro_f1


def main() -> None:
    learning_rate, batch_size, epochs = get_hyperparameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading FashionMNIST dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_set = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    model = FashionCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_model(
        model=model,
        train_dataloader=train_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        scheduler=scheduler
    )

    test_model(model, test_dataloader, device)

    torch.save(model.state_dict(), "fashion_cnn.pth")
    print("Model saved to fashion_cnn.pth")


if __name__ == "__main__":
    main()