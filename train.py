"""
train.py: Multi-Class Training Pipeline for Hybrid QNN (Digits 0-9)
Utilizes differential layer learning rates to achieve high validation accuracy (>80%).
Saves final trained model weights to model_multiclass.pth.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import HybridQNN


def get_mnist_loaders(data_dir="./data", batch_size=32, subset_train=8000, subset_test=1000):
    """
    Downloads and prepares full MNIST dataset (10 classes: 0 through 9).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("--> Loading full MNIST dataset (Digits 0-9)...")
    full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    full_test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    if subset_train and subset_train > 0 and subset_train < len(full_train):
        train_indices = list(range(subset_train))
        train_dataset = Subset(full_train, train_indices)
    else:
        train_dataset = full_train

    if subset_test and subset_test > 0 and subset_test < len(full_test):
        test_indices = list(range(subset_test))
        test_dataset = Subset(full_test, test_indices)
    else:
        test_dataset = full_test

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"--> Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples (10 Classes).")
    return train_loader, test_loader


def evaluate(model, test_loader, criterion, device):
    """Computes average loss and top-1 accuracy on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images, return_probs=False)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


def train():
    parser = argparse.ArgumentParser(description="Train Multi-Class Hybrid Quantum-Classical Classifier")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr-cnn", type=float, default=0.001, help="Learning rate for CNN extractor (default: 0.001)")
    parser.add_argument("--lr-quantum", type=float, default=0.01, help="Learning rate for QNode & Head (default: 0.01)")
    parser.add_argument("--subset-train", type=int, default=8000, help="Training subset size (default: 8000, 0 for all)")
    parser.add_argument("--subset-test", type=int, default=1000, help="Evaluation subset size (default: 1000, 0 for all)")
    parser.add_argument("--save-path", type=str, default="model_multiclass.pth", help="Path to save model weights")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to cache MNIST data")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"--> Using compute device: {device}")

    # Prepare DataLoaders
    train_loader, test_loader = get_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        subset_train=args.subset_train,
        subset_test=args.subset_test
    )

    # Initialize 10-class Hybrid QNN
    print("--> Initializing 10-Class Hybrid Quantum-Classical Neural Network...")
    model = HybridQNN(n_classes=10).to(device)

    # Differential Learning Rates: Stable CNN representation + rapid Quantum convergence
    param_groups = [
        {"params": model.conv1.parameters(), "lr": args.lr_cnn},
        {"params": model.conv2.parameters(), "lr": args.lr_cnn},
        {"params": model.fc_reduction.parameters(), "lr": args.lr_cnn},
        {"params": model.qlayer.parameters(), "lr": args.lr_quantum},
        {"params": model.fc_out.parameters(), "lr": args.lr_quantum},
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(param_groups)

    print("\n" + "=" * 70)
    print(f"Multi-Class Training: {args.epochs} Epochs | Batch Size: {args.batch_size}")
    print(f"LR (CNN): {args.lr_cnn} | LR (Quantum + Head): {args.lr_quantum}")
    print("=" * 70)

    start_total_time = time.time()
    best_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_start_time = time.time()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images, return_probs=False)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

            # In-batch progress logging
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (correct_train / total_train) * 100.0
                print(f"  [Epoch {epoch}/{args.epochs}] Step [{batch_idx + 1}/{len(train_loader)}] "
                      f"Batch Loss: {loss.item():.4f} | Running Train Acc: {batch_acc:.2f}%")

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        epoch_duration = time.time() - epoch_start_time

        print(f"--> [Epoch {epoch}/{args.epochs} Complete ({epoch_duration:.1f}s)] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        print("-" * 70)

        # Checkpoint if best accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  [Checkpoint] New Best Model Saved (Accuracy: {best_accuracy:.2f}%)")

    total_training_time = time.time() - start_total_time
    print(f"\nMulti-Class Training completed in {total_training_time:.2f} seconds.")
    print(f"--> Best Test Accuracy Achieved: {best_accuracy:.2f}%")
    print(f"--> Saved trained weights to: {os.path.abspath(args.save_path)}")


if __name__ == "__main__":
    train()
