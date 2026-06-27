"""
Streamlit demo for the astronomical object classifier.

Run locally:
    streamlit run app.py

Set CHECKPOINT_PATH env var to point at a non-default checkpoint:
    CHECKPOINT_PATH=checkpoints/baseline_resnet50.pt streamlit run app.py

For deployment notes see README.md.
"""

import json as _json
import os
import re
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import torch
from PIL import Image

from inference import build_eval_transform, overlay_heatmap, predict
from model import DEFAULT_CLASSES, load_checkpoint
from ood_detection import score_image as ood_score_image


# Pretty display name → folder-style class name produced by training.
CLASS_DISPLAY = {
    "spiral_galaxy": "Spiral galaxy",
    "elliptical_galaxy": "Elliptical galaxy",
    "nebula": "Nebula",
    "star_cluster": "Star cluster",
    "planetary_object": "Planetary object",
}

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "checkpoints/baseline_resnet50.pt",
)

# ---------- fetch checkpoint from Google Drive (Streamlit Cloud) ----------

def _extract_drive_id(value: str) -> str:
    """Accept either a bare Drive ID or a full share URL and return the ID."""
    value = value.strip()
    # …/folders/<ID>?…  or  …/file/d/<ID>/…  or  …?id=<ID>
    m = re.search(r"/(?:folders|d)/([A-Za-z0-9_-]+)", value)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    if m:
        return m.group(1)
    return value  # already a bare ID


@st.cache_resource(show_spinner="Downloading model from Google Drive…")
def ensure_checkpoint(path: str) -> tuple[bool, str]:
    """Download the checkpoint from a shared Drive folder on first run.

    Streamlit Cloud has an ephemeral filesystem and no `.pt` is committed to
    the repo, so we pull the `checkpoints/` contents (the `.pt` plus
    `ood_thresholds.json`) from Google Drive once per container.

    Set the Drive *folder* ID (or its share URL) in the app's
    **Settings → Secrets** as `GDRIVE_FOLDER_ID`, or via the env var of the
    same name. The folder must be shared as "Anyone with the link".
    Returns (ok, message) where message explains any failure.
    """
    if Path(path).is_file():
        return True, "already present"

    raw = st.secrets.get("GDRIVE_FOLDER_ID", os.environ.get("GDRIVE_FOLDER_ID"))
    if not raw:
        return False, "GDRIVE_FOLDER_ID is not set in Secrets."

    folder_id = _extract_drive_id(str(raw))
    out_dir = Path(path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        gdown.download_folder(
            id=folder_id,
            output=str(out_dir),
            quiet=False,
            use_cookies=False,
        )
    except Exception as exc:  # surface the real reason instead of "no model"
        return False, f"gdown failed: {exc}"

    # gdown often nests files under a subdir named after the remote folder.
    # Flatten: move any matching files we find up into out_dir.
    target = Path(path)
    if not target.is_file():
        for found in out_dir.rglob(target.name):
            found.replace(target)
            break
    for found in out_dir.rglob("ood_thresholds.json"):
        dest = out_dir / "ood_thresholds.json"
        if found != dest:
            found.replace(dest)
        break

    if target.is_file():
        return True, "downloaded"
    listing = ", ".join(p.name for p in out_dir.rglob("*") if p.is_file()) or "(empty)"
    return False, (
        f"download finished but {target.name!r} not found. "
        f"Files pulled: {listing}. Check the folder is shared 'Anyone with the link'."
    )


# ---------- model loading (cached) ----------

@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(path: str):
    """Cached because Streamlit re-runs the script on every interaction."""
    if not Path(path).is_file():
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size, mean, std = load_checkpoint(path, device)

    # OOD thresholds live next to the checkpoint (written by notebook §14.6).
    # Fall back to uncalibrated defaults if calibration hasn't been run.
    thresh_path = Path(path).parent / "ood_thresholds.json"
    if thresh_path.is_file():
        with open(thresh_path) as f:
            ood_thresholds = _json.load(f)
    else:
        ood_thresholds = {"msp_threshold": 0.5, "energy_threshold": -5.0}

    return {
        "model": model,
        "classes": classes,
        "image_size": image_size,
        "mean": mean,
        "std": std,
        "device": device,
        "ood_thresholds": ood_thresholds,
    }


# ---------- UI ----------

st.set_page_config(
    page_title="Astronomical Object Classifier",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 Astronomical Object Classifier")
st.markdown(
    "Upload an astronomical image and the model will predict which of five "
    "categories it belongs to: **spiral galaxy, elliptical galaxy, nebula, "
    "star cluster, or planetary object**. "
    "The heatmap shows which regions of the image most influenced the prediction."
)

# ----- Sidebar -----
with st.sidebar:
    st.header("About")
    st.markdown(
        "**Architecture.** ResNet50 pretrained on ImageNet, fine-tuned on a "
        "composite dataset of ~5,000 astronomical images.\n\n"
        "**Explanation.** Heatmaps generated with Grad-CAM (Selvaraju et al., 2017)."
    )

    st.divider()
    st.header("Settings")
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    overlay_alpha = st.slider(
        "Heatmap opacity", 0.0, 1.0, 0.45, 0.05,
        disabled=not show_gradcam,
    )

    st.divider()
    ok, msg = ensure_checkpoint(CHECKPOINT_PATH)
    bundle = load_model_cached(CHECKPOINT_PATH)
    if bundle is not None:
        st.success(f"Model loaded on **{bundle['device']}**")
        st.caption(f"Checkpoint: `{CHECKPOINT_PATH}`")
        st.caption(f"Classes: {', '.join(bundle['classes'])}")
    else:
        st.error("No checkpoint found")
        st.caption(f"Looked at: `{CHECKPOINT_PATH}`")
        st.caption(f"Download status: {msg}")


# ----- Main area -----
if bundle is None:
    st.warning(
        "**No trained model is available.** Train the model in the notebook "
        "(`astro_classifier_training.ipynb`) and copy the saved checkpoint to "
        f"`{CHECKPOINT_PATH}` — or point `CHECKPOINT_PATH` at a different file."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload an astronomical image (JPG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("👆 Upload an image to get started.")
    st.stop()

# Run inference
try:
    image = Image.open(uploaded)
except Exception as e:
    st.error(f"Could not read that file as an image: {e}")
    st.stop()

with st.spinner("Classifying…"):
    result = predict(
        image=image,
        model=bundle["model"],
        classes=bundle["classes"],
        image_size=bundle["image_size"],
        mean=bundle["mean"],
        std=bundle["std"],
        device=bundle["device"],
        compute_gradcam=show_gradcam,
    )

    # Out-of-distribution score: does this image even look astronomical?
    # Reuses the exact eval transform predict() uses, so scoring is consistent.
    ood_tf = build_eval_transform(bundle["image_size"], bundle["mean"], bundle["std"])
    ood_result = ood_score_image(
        image,
        bundle["model"],
        ood_tf,
        bundle["device"],
        msp_threshold=bundle["ood_thresholds"]["msp_threshold"],
        energy_threshold=bundle["ood_thresholds"]["energy_threshold"],
    )

# ----- Results layout -----
left, right = st.columns([1, 1])

with left:
    st.subheader("Input")
    st.image(image, use_container_width=True)

    if show_gradcam and result.heatmap is not None:
        st.subheader("Grad-CAM overlay")
        overlay = overlay_heatmap(result.input_image, result.heatmap, alpha=overlay_alpha)
        st.image(overlay, use_container_width=True)
        st.caption(
            "Bright regions = strongest evidence for the predicted class. "
            "If the highlight is on background or borders, the model may be "
            "relying on a spurious feature."
        )

with right:
    st.subheader("Prediction")

    # OOD gate: warn before showing the class if the image looks non-astronomical.
    # Energy scoring is the primary flag (higher AUROC than MSP on this model).
    if ood_result.energy_is_ood:
        st.warning(
            "⚠ **This image may not be an astronomical object.** "
            f"OOD score (energy): {ood_result.energy_score:.2f} "
            f"(threshold: {bundle['ood_thresholds']['energy_threshold']:.2f}). "
            "The prediction below should be treated as low-confidence."
        )

    display_name = CLASS_DISPLAY.get(result.predicted_class, result.predicted_class)
    st.markdown(f"### {display_name}")
    st.progress(result.confidence, text=f"{result.confidence * 100:.1f}% confidence")

    with st.expander("OOD detection details"):
        st.metric(
            "MSP score", f"{ood_result.msp_score:.3f}",
            delta=f"{ood_result.msp_score - bundle['ood_thresholds']['msp_threshold']:+.3f} vs threshold",
        )
        st.metric(
            "Energy score", f"{ood_result.energy_score:.3f}",
            delta=f"{ood_result.energy_score - bundle['ood_thresholds']['energy_threshold']:+.3f} vs threshold",
        )
        st.caption(
            "Higher scores indicate the image is less similar to the training "
            "distribution. Both methods use the existing classifier — no extra model."
        )

    st.subheader("All class probabilities")
    # Sort descending so the predicted class is at top.
    rows = sorted(
        zip(bundle["classes"], result.probabilities),
        key=lambda x: -x[1],
    )
    for cls, prob in rows:
        name = CLASS_DISPLAY.get(cls, cls)
        col_a, col_b = st.columns([2, 5])
        col_a.write(f"**{name}**")
        col_b.progress(prob, text=f"{prob * 100:.1f}%")

    # Friendly nudge when the model isn't very sure.
    if result.confidence < 0.5:
        st.warning(
            "Confidence is low. The image might not clearly match any of the "
            "five classes the model was trained on, or it could be visually "
            "ambiguous between two classes (e.g. dense star cluster vs galaxy core)."
        )
    elif result.confidence > 0.95 and show_gradcam:
        st.info(
            "Very high confidence. Check the heatmap to confirm the model "
            "looked at the actual object and not a background feature."
        )
