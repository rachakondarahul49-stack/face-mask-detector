"""
Train the face mask classifier.

Dataset: https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
Expects an ImageFolder-style layout:
    data/
        with_mask/
        without_mask/

Usage:
    python src/train.py --data-dir data --epochs 10 --out models/face_mask_model.pth
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import build_model, CLASSES

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Path to ImageFolder-style dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--out", default="models/face_mask_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = datasets.ImageFolder(args.data_dir, transform=TRANSFORM)
    assert dataset.classes == CLASSES, (
        f"Expected class folders {CLASSES}, found {dataset.classes}. "
        "Rename your folders or update CLASSES in model.py."
    )

    val_size = int(args.val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Classes: {dataset.classes}")
    print(f"Train: {train_size} | Val: {val_size}")

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_acc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {running_loss / len(train_loader):.3f} | "
            f"Train acc: {train_acc:.2f}% | Val acc: {val_acc:.2f}%"
        )

        # Save the BEST checkpoint by validation accuracy, not just the last epoch.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> New best model saved to {args.out}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
