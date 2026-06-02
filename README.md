# Astronomical Object Classifier

Deep-learning classifier for astronomical images, built as a multi-stage computer-vision project. Distinguishes five categories of celestial objects from raw image input, with Grad-CAM heatmaps explaining each prediction and out-of-distribution detection that flags non-astronomical uploads.

**Classes:** spiral galaxy, elliptical galaxy, nebula, star cluster, planetary object.

---

## Results

ResNet50 baseline on the held-out test set (346 images):

| Metric | Value |
|---|---|
| Test accuracy | **96.2%** |
| Macro F1 | 0.964 |
| Inference time (GPU) | ~98 ms/image |
| Inference time (CPU) | ~67 ms/image |
| Total parameters | 23.5M (ResNet50) |
| Training time | ~35 min on an RTX 4050 (2-phase) |

Per-class metrics, the confusion matrix, and the Grad-CAM analysis are produced in the training notebook (sections 8–13).

### Architecture comparison

Three backbones trained under an identical protocol (same splits, augmentations, epochs, optimizer). Latencies are single-image (batch = 1), averaged over 50 runs:

| Model | Params | Test acc | Macro F1 | GPU latency | CPU latency |
|---|---|---|---|---|---|
| ResNet50 (baseline) | 23.5M | 96.2% | 0.964 | 98 ms | 67 ms |
| EfficientNet-B0 | 4.0M | 94.5% | 0.948 | 11 ms | 35 ms |
| ViT-B/16 | 85.8M | 96.5% | 0.962 | 71 ms | 128 ms |

ViT-B/16 edges out the others on accuracy, EfficientNet-B0 is by far the most efficient, and ResNet50 is the balanced baseline shipped in the demo.

---

## Quick demo

```bash
git clone https://github.com/Pranav-Ram-R/astro-classifier.git
cd astro-classifier
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

Open the URL it prints (usually http://localhost:8501) and upload an astronomical image. The app shows the predicted class, confidence, full probability distribution, a Grad-CAM heatmap overlay indicating which image regions drove the prediction, and an out-of-distribution flag for inputs that don't look astronomical.

If you don't have a trained checkpoint yet, see the **Reproduce from scratch** section below.

---

## Project structure

```
astro-classifier/
├── README.md                        this file
├── astro_classifier_training.ipynb  training, comparison, ablation, Grad-CAM, OOD (sections 1–14)
├── ood_detection.py                 OOD scoring (MSP + energy) imported by the notebook
│
├── data_collection/                 stage 1 — assemble the dataset
│   ├── 01_download_galaxy10.py            spiral + elliptical galaxies
│   ├── 02_scrape_hubble.py                nebulae from ESA/Hubble
│   ├── 02b_scrape_hubble_starclusters.py  star clusters from ESA/Hubble
│   ├── 03_fetch_nasa_planets.py           solar-system planets from NASA Image API
│   ├── 04_inspect_data.py                 counts + sample-grid QA
│   ├── requirements.txt
│   └── README.md
│
├── demo/                            stage 3 — Streamlit interface
│   ├── app.py                       Streamlit UI (prediction + Grad-CAM + OOD flag)
│   ├── model.py                     ResNet50 factory + checkpoint loader
│   ├── inference.py                 preprocessing + end-to-end predict()
│   ├── gradcam.py                   Grad-CAM as a context manager
│   ├── ood_detection.py             OOD scoring (demo copy)
│   ├── requirements.txt
│   └── README.md
│
├── checkpoints/                     trained model artifacts (gitignored)
│   └── baseline_resnet50.pt
│
└── data/                            dataset (gitignored)
    └── processed/
        ├── spiral_galaxy/
        ├── elliptical_galaxy/
        ├── nebula/
        ├── star_cluster/
        └── planetary_object/
```

---

## Reproduce from scratch

Three stages: collect data → train → evaluate / demo. Each is self-contained.

### Stage 1 — Collect the dataset

```bash
cd data_collection
pip install -r requirements.txt
python 01_download_galaxy10.py          # ~2.5 GB download, then a few minutes
python 02_scrape_hubble.py              # ~30-45 min, rate-limited to be polite
python 02b_scrape_hubble_starclusters.py
python 03_fetch_nasa_planets.py         # ~10-15 min
python 04_inspect_data.py               # writes a sample-grid QA image
```

After running, open `data/sample_grid.png` and confirm each class folder visually matches its label. The NASA planetary results contain some non-planet content (mission patches, illustrations); ~10–25% manual deletion is normal.

Full details, troubleshooting, and per-class targets are in [`data_collection/README.md`](data_collection/README.md).

### Stage 2 — Train the model

Open `astro_classifier_training.ipynb` and run all cells top to bottom. It runs on a local CUDA GPU (developed on an RTX 4050) or on Google Colab — set the runtime to GPU there (Runtime → Change runtime type → T4 GPU). The notebook auto-detects Colab and adjusts paths.

The notebook handles:
- loading data via `ImageFolder` with a stratified train/val/test split
- majority-class subsampling so the imbalance is ~2.7:1
- two-phase fine-tuning (frozen backbone for 3 epochs, then full fine-tune for 12)
- mixed-precision training (AMP)
- per-epoch checkpointing with resume-from-checkpoint support
- test-set evaluation with confusion matrix + per-class report
- Grad-CAM visualization, an EfficientNet/ViT comparison, a class-weighting ablation, and OOD detection

Training the baseline takes ~35 minutes on an RTX 4050. The final checkpoint is saved to `checkpoints/baseline_resnet50.pt`.

### Stage 3 — Run the demo

```bash
cd demo
pip install -r requirements.txt
streamlit run app.py
```

Deploy to Streamlit Community Cloud (free): see [`demo/README.md`](demo/README.md) for the two checkpoint-hosting strategies (Git LFS vs HuggingFace download on cold start).

---

## How it works

**Architecture.** ResNet50 (He et al., 2015) pretrained on ImageNet-1K, with the 2048→1000 final layer replaced by a new 2048→5 head for the project classes.

**Transfer learning recipe.** Two phases. Phase 1 freezes the backbone and trains only the new head at LR 1e-3 for 3 epochs — this anchors the new classifier without disturbing the pretrained features. Phase 2 unfreezes everything and fine-tunes at LR 1e-4 with cosine annealing for 12 epochs.

**Class imbalance.** Cross-entropy loss is weighted by inverse class frequency, computed automatically from training counts. An ablation (notebook section 12) compares per-class F1 with and without this weighting.

**Explainability.** Grad-CAM (Selvaraju et al., 2017) computed on `model.layer4` (the final convolutional stage, 7×7×2048). The same `GradCAM` class is reused by both the training notebook and the demo app.

**Out-of-distribution detection.** Two training-free scores — maximum softmax probability (MSP) and energy (`-logsumexp(logits)`) — flag inputs that don't resemble the training data. A threshold calibrated at 95% true-positive rate is exported to `ood_thresholds.json` and consumed by the demo to warn on non-astronomical uploads.

---

## Hardware requirements

| Workload | Minimum | Recommended |
|---|---|---|
| Inference (demo) | CPU (~0.07s/image) | Any modern GPU |
| Training | NVIDIA T4 (free Colab) | RTX 4050 / T4 / L4 / A100 |
| Disk space | 5 GB (data + model) | 10 GB |

---

## Acknowledgments and data attribution

- **Galaxy10 DECaLS** — Henry Leung, [henrysky/Galaxy10](https://github.com/henrysky/Galaxy10). Underlying data: Galaxy Zoo (Lintott et al., 2008) and DESI Legacy Imaging Surveys.
- **ESA/Hubble image archive** — NASA, ESA, and the Hubble Heritage Team. Public domain under [esahubble.org/copyright](https://esahubble.org/copyright/).
- **NASA Image and Video Library** — NASA. Public domain under [NASA media usage guidelines](https://www.nasa.gov/multimedia/guidelines/index.html).
- **ResNet50 pretrained weights** — torchvision IMAGENET1K_V2.
- **Stakeholders for this project** — Janil Jain, Jaskirat Singh Maskeen, Priyal Keswani.

---

## License

Source code: MIT.
Trained model weights: distributed under the same terms as the underlying datasets (public, attribution requested for academic use).
