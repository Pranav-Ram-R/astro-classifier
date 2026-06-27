"""
End-to-end inference helpers: preprocess a PIL image, run the model,
optionally compute Grad-CAM, return a single result object.

The Streamlit app and any future batch/CLI script both go through `predict()`
so behavior is consistent.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gradcam import GradCAM


@dataclass
class Prediction:
    """One image's prediction with all info the UI might want to render."""
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: List[float]      # per-class probability, indexed like `classes`
    heatmap: Optional[np.ndarray]   # HxW in [0,1], or None if grad-cam was skipped
    input_image: np.ndarray         # HxWx3 in [0,1], the preprocessed (resized) input


def build_eval_transform(image_size: int, mean: List[float], std: List[float]):
    """Deterministic preprocessing: resize → center-crop → tensor → normalize."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),  # keep the standard 256→224 ratio
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """Invert the normalization step so we can display the preprocessed image."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap onto an HxWx3 [0,1] image using the jet colormap.

    matplotlib is imported lazily so the inference module doesn't pull in the
    plotting stack when callers don't need overlays.
    """
    # matplotlib.cm.get_cmap was removed in matplotlib 3.9; use the
    # colormaps registry, falling back to the old API on older versions.
    try:
        import matplotlib
        cmap = matplotlib.colormaps["jet"]
    except (ImportError, AttributeError, KeyError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]   # drop alpha channel
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)


def predict(
    image: Image.Image,
    model: torch.nn.Module,
    classes: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    device: torch.device,
    compute_gradcam: bool = True,
) -> Prediction:
    """Run end-to-end inference. Returns a `Prediction` dataclass.

    Always runs the forward pass once; if `compute_gradcam` is True it runs an
    additional backward pass for the heatmap.
    """
    image = image.convert("RGB")
    tf = build_eval_transform(image_size, mean, std)
    input_tensor = tf(image).unsqueeze(0).to(device)
    input_image = denormalize(input_tensor[0], mean, std)

    if compute_gradcam:
        # ResNet50's last conv stage. Change this for other backbones.
        with GradCAM(model, model.layer4) as cam:
            heatmap, pred_idx, conf = cam(input_tensor)
        # Recompute the full probability vector in inference mode.
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().tolist()
    else:
        heatmap = None
        with torch.no_grad():
            logits = model(input_tensor)
            probs_t = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs_t.argmax().item())
            conf = float(probs_t[pred_idx].item())
            probs = probs_t.cpu().tolist()

    return Prediction(
        predicted_class=classes[pred_idx],
        predicted_index=pred_idx,
        confidence=conf,
        probabilities=probs,
        heatmap=heatmap,
        input_image=input_image,
    )
