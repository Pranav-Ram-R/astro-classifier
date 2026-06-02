"""
Sanity-check the assembled dataset.

Prints per-class file counts and writes a sample grid PNG so you can
visually confirm each class folder contains what it should.

Usage:
  python 04_inspect_data.py
  python 04_inspect_data.py --samples 8 --out report.png
"""

import argparse
import glob
import os
import random
import sys

CLASSES = [
    "spiral_galaxy",
    "elliptical_galaxy",
    "nebula",
    "star_cluster",
    "planetary_object",
]


def count_files(data_dir):
    counts = {}
    for c in CLASSES:
        path = os.path.join(data_dir, c)
        if not os.path.isdir(path):
            counts[c] = 0
            continue
        files = glob.glob(os.path.join(path, "*.jpg")) + \
                glob.glob(os.path.join(path, "*.jpeg")) + \
                glob.glob(os.path.join(path, "*.png"))
        counts[c] = len(files)
    return counts


def print_counts(counts):
    print(f"{'Class':<22} {'Count':>8}")
    print("-" * 32)
    for c, n in counts.items():
        flag = " " if n >= 300 else " ⚠ low"
        print(f"{c:<22} {n:>8}{flag}")
    print("-" * 32)
    total = sum(counts.values())
    print(f"{'TOTAL':<22} {total:>8}")

    if total == 0:
        print("\nNo images found. Did the download scripts run successfully?")
        return

    max_n = max(counts.values())
    min_n = min(n for n in counts.values() if n > 0) if any(counts.values()) else 0
    if min_n and max_n / min_n > 5:
        print(f"\n⚠ class imbalance: largest is {max_n / min_n:.1f}x the smallest.")
        print("  consider downsampling the largest class or topping up the smallest.")


def make_grid(data_dir, samples, out_path):
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("\nmatplotlib and/or Pillow not installed. Skipping sample grid.")
        print("Install with: pip install matplotlib Pillow")
        return

    random.seed(42)
    fig, axes = plt.subplots(
        len(CLASSES), samples,
        figsize=(samples * 2.2, len(CLASSES) * 2.2),
    )
    if len(CLASSES) == 1:
        axes = [axes]

    for row, c in enumerate(CLASSES):
        path = os.path.join(data_dir, c)
        files = []
        if os.path.isdir(path):
            files = (glob.glob(os.path.join(path, "*.jpg"))
                     + glob.glob(os.path.join(path, "*.jpeg"))
                     + glob.glob(os.path.join(path, "*.png")))

        picks = random.sample(files, min(samples, len(files))) if files else []

        for col in range(samples):
            ax = axes[row][col] if samples > 1 else axes[row]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < len(picks):
                try:
                    img = Image.open(picks[col]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "load err", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                        color="gray", fontsize=8)
            if col == 0:
                ax.set_ylabel(c, rotation=0, ha="right", va="center",
                              fontsize=10, labelpad=40)

    plt.suptitle("Dataset sample grid", fontsize=12)
    plt.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close()
    print(f"\nSample grid saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="data/processed",
        help="Directory containing class subfolders",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Sample images to display per class (default: 5)",
    )
    parser.add_argument(
        "--out", default="data/sample_grid.png",
        help="Output path for the sample grid PNG",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"ERROR: data dir does not exist: {args.data}", file=sys.stderr)
        sys.exit(1)

    counts = count_files(args.data)
    print_counts(counts)
    make_grid(args.data, args.samples, args.out)


if __name__ == "__main__":
    main()
