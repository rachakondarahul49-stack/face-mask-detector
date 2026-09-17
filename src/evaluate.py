"""
Evaluate a trained model on a held-out set: accuracy, precision, recall,
F1-score and a confusion matrix.

Usage:
    python src/evaluate.py --data-dir data --weights models/face_mask_model.pth
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import load_model, CLASSES

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Held-out eval set (ImageFolder layout)")
    parser.add_argument("--weights", default="models/face_mask_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    dataset = datasets.ImageFolder(args.data_dir, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
