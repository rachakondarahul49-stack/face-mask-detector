"""
Real-time webcam face mask detection.

Note: this classifies the whole frame, not an individually detected face.
For multi-face detection in one frame, pair this with a face detector
(e.g. OpenCV's Haar cascade or a DNN face detector) and run the classifier
on each cropped face region.

Usage:
    python src/webcam_demo.py --weights models/face_mask_model.pth
"""
import argparse

import cv2
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
    parser.add_argument("--weights", default="models/face_mask_model.pth")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = TRANSFORM(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = int(probs.argmax().item())
            label = CLASSES[pred_idx]
            confidence = probs[pred_idx].item() * 100

        color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
        text = f"{label} ({confidence:.1f}%)"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 3)
        cv2.imshow("Face Mask Detector - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
