"""
Run mask / no-mask prediction on a single image.

Usage:
    python src/predict.py --image path/to/face.jpg --weights models/face_mask_model.pth
"""
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import load_model, CLASSES

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="models/face_mask_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    img = Image.open(args.image).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = int(probs.argmax().item())

    print(f"Prediction: {CLASSES[pred_idx]} ({probs[pred_idx].item() * 100:.1f}% confidence)")


if __name__ == "__main__":
    main()
