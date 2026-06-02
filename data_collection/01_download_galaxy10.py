"""
Download Galaxy10 DECaLS and extract spiral / elliptical galaxies as JPEGs.

Galaxy10 DECaLS: 17,736 colored galaxy images, 256x256px, 10 morphology classes.
Source: https://github.com/henrysky/Galaxy10

The dataset is downloaded by astroNN to ~/.astroNN/datasets/ on first run
(~2.54 GB, plan for 10-20 minutes on a decent connection).

Class mapping:
  Spiral Galaxy     <- Galaxy10 classes 5, 6, 7
                       (barred, unbarred tight, unbarred loose spiral)
  Elliptical Galaxy <- Galaxy10 classes 2, 3, 4
                       (round smooth, in-between, cigar-shaped)

Usage:
  python 01_download_galaxy10.py
  python 01_download_galaxy10.py --per_class 2000
  python 01_download_galaxy10.py --output /custom/path
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Galaxy10 class IDs we want to merge into our two output classes.
SPIRAL_CLASSES = [5, 6, 7]
ELLIPTICAL_CLASSES = [2, 3, 4]


def load_galaxy10():
    """Lazy import — astroNN is heavy and only needed here."""
    try:
        from astroNN.datasets import load_galaxy10 as _loader
    except ImportError:
        print("ERROR: astroNN is not installed.", file=sys.stderr)
        print("Run: pip install astroNN", file=sys.stderr)
        sys.exit(1)

    print("Loading Galaxy10 DECaLS (downloads ~2.54 GB on first run)...")
    images, labels = _loader()
    print(f"  loaded {len(images)} images, shape {images.shape[1:]}\n")
    return images, labels


def extract(images, labels, source_classes, target_dir, target_count, name):
    """Save up to `target_count` images matching `source_classes` as JPEGs."""
    os.makedirs(target_dir, exist_ok=True)

    mask = np.isin(labels, source_classes)
    indices = np.where(mask)[0]
    print(f"{name}: {len(indices)} images in Galaxy10 classes {source_classes}")

    if len(indices) > target_count:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(indices, size=target_count, replace=False)
        print(f"  downsampling to {target_count}")

    for i, idx in enumerate(indices):
        img = Image.fromarray(images[idx])
        out_path = os.path.join(target_dir, f"{name}_{i:05d}.jpg")
        img.save(out_path, "JPEG", quality=92)
        if (i + 1) % 250 == 0:
            print(f"  saved {i + 1}/{len(indices)}")

    print(f"  done -> {target_dir}\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--per_class", type=int, default=1500,
        help="Max images per output class (default: 1500)",
    )
    parser.add_argument(
        "--output", default="data/processed",
        help="Output base directory (default: data/processed)",
    )
    args = parser.parse_args()

    images, labels = load_galaxy10()

    extract(
        images, labels, SPIRAL_CLASSES,
        os.path.join(args.output, "spiral_galaxy"),
        args.per_class, "spiral_galaxy",
    )
    extract(
        images, labels, ELLIPTICAL_CLASSES,
        os.path.join(args.output, "elliptical_galaxy"),
        args.per_class, "elliptical_galaxy",
    )

    print("Galaxy10 extraction complete.")


if __name__ == "__main__":
    main()
