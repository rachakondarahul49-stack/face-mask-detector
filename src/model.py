"""Model definition for the face mask classifier."""
import torch
import torch.nn as nn
from torchvision import models

CLASSES = ["with_mask", "without_mask"]


def build_model(pretrained: bool = True) -> nn.Module:
    """Build a MobileNetV2 classifier with a 2-class head.

    Args:
        pretrained: if True, initialize the backbone with ImageNet weights.
    """
    weights = "DEFAULT" if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    return model


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    """Load a trained model from a saved state_dict, ready for inference."""
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
