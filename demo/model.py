"""
Model factory + checkpoint loader for the astronomical object classifier demo.

The training notebook saves checkpoints with this schema:
    {
        'model_state':   state_dict for ResNet50 with 5-class head,
        'classes':       list of class names in label-index order,
        'image_size':    e.g. 224,
        'imagenet_mean': [0.485, 0.456, 0.406],
        'imagenet_std':  [0.229, 0.224, 0.225],
        'test_accuracy': float (optional),
    }

This module rebuilds the architecture and loads those weights.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models


DEFAULT_CLASSES = [
    "elliptical_galaxy",
    "nebula",
    "planetary_object",
    "spiral_galaxy",
    "star_cluster",
]


def build_model(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    """Construct the ResNet50 architecture with a `num_classes`-way head.

    `pretrained=False` here because we'll load the user's fine-tuned weights
    on top — fetching ImageNet weights at app boot is wasted bandwidth.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, List[str], int, List[float], List[float]]:
    """Load a saved checkpoint and return (model, classes, image_size, mean, std).

    Falls back to ImageNet defaults for mean/std/image_size if the checkpoint
    pre-dates those fields being saved.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    classes = ckpt.get("classes", DEFAULT_CLASSES)
    image_size = ckpt.get("image_size", 224)
    imagenet_mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    imagenet_std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    return model, classes, image_size, imagenet_mean, imagenet_std
