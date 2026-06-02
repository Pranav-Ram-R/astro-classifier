"""
Out-of-distribution (OOD) detection for the astronomical classifier.

Implements two scoring methods:

1. Maximum Softmax Probability (MSP) — Hendrycks & Gimpel 2017.
   The simplest possible baseline. OOD score is just (1 - max softmax probability).
   Intuition: in-distribution images get confident predictions (one class
   close to 1.0), out-of-distribution images get diffuse predictions
   (probabilities spread across multiple classes).

2. Energy-based scoring — Liu et al. 2020 (NeurIPS).
   Uses the log-sum-exp of the logits as a confidence score. In-distribution
   images have a large negative "free energy," OOD images have a small one.
   Empirically beats MSP on most benchmarks with no extra training.

Both methods require no retraining and no additional data. They just look at
the existing model's logits differently.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class OODResult:
    """OOD scoring output. Higher score = more likely to be out-of-distribution."""
    msp_score: float             # in [0, 1]: 1 - max softmax probability
    energy_score: float          # negative log-sum-exp of logits; range ~[-40, 0]
    msp_is_ood: bool             # MSP score exceeds threshold
    energy_is_ood: bool          # energy score exceeds threshold


# Defaults — calibrate to your in-distribution validation set; see calibrate().
DEFAULT_MSP_THRESHOLD = 0.5
DEFAULT_ENERGY_THRESHOLD = -5.0


def msp_score(logits: torch.Tensor) -> torch.Tensor:
    """Maximum-softmax-probability OOD score. Shape: (batch,)."""
    probs = F.softmax(logits, dim=-1)
    max_p, _ = probs.max(dim=-1)
    return 1.0 - max_p


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy-based OOD score. Higher = more likely OOD. Shape: (batch,)."""
    # Free energy = -T * logsumexp(logits / T). We negate so higher = more OOD.
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


@torch.no_grad()
def score_image(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    msp_threshold: float = DEFAULT_MSP_THRESHOLD,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
) -> OODResult:
    """Score a single image. Returns both MSP and energy scores plus binary OOD flags."""
    img_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    logits = model(img_tensor)

    msp = float(msp_score(logits).item())
    energy = float(energy_score(logits).item())

    return OODResult(
        msp_score=msp,
        energy_score=energy,
        msp_is_ood=msp > msp_threshold,
        energy_is_ood=energy > energy_threshold,
    )


@torch.no_grad()
def score_batch(
    images: List[Image.Image],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Score a list of images. Returns dict with msp_scores and energy_scores arrays."""
    model.eval()
    msp_scores, energy_scores = [], []
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        tensors = torch.stack([transform(img.convert("RGB")) for img in batch_imgs]).to(device)
        logits = model(tensors)
        msp_scores.extend(msp_score(logits).cpu().tolist())
        energy_scores.extend(energy_score(logits).cpu().tolist())
    return {
        "msp_scores": np.array(msp_scores),
        "energy_scores": np.array(energy_scores),
    }


def calibrate_thresholds(
    in_dist_scores: dict,
    target_tpr: float = 0.95,
) -> dict:
    """Pick thresholds that keep target_tpr of in-distribution data classified as ID.

    Common choice: target_tpr = 0.95, meaning we accept 5% false positives
    (in-dist images incorrectly flagged as OOD) in exchange for catching as
    much OOD as possible at that operating point.
    """
    msp_threshold = float(np.quantile(in_dist_scores["msp_scores"], target_tpr))
    energy_threshold = float(np.quantile(in_dist_scores["energy_scores"], target_tpr))
    return {
        "msp_threshold": msp_threshold,
        "energy_threshold": energy_threshold,
        "target_tpr": target_tpr,
    }


def compute_auroc(in_dist_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Area under ROC curve for separating in-dist from OOD by score.

    Higher OOD scores should correspond to OOD samples. AUROC of 1.0 = perfect
    separation, 0.5 = random.
    """
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.zeros(len(in_dist_scores)),
                              np.ones(len(ood_scores))])
    scores = np.concatenate([in_dist_scores, ood_scores])
    return float(roc_auc_score(labels, scores))
