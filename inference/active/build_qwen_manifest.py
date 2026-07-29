"""
build_qwen_manifest.py

Samples N (image, ground_truth) pairs from a Urdu_Word_Dataset-style split
(images/ folder + gt_txt/ folder with one .txt per image, produced by
data_preprocessing/gt_preprocessing.py) and writes them to a CSV in the
`image_path,ground_truth` format qwen_eval.py's evaluate() expects.

Designed for the CALIBRATION step specifically: point it at REAL validation
images (not DiffusionRec-generated ones) first, so the resulting manifest
gives you a trustworthy check against the paper's 74.40% / 25.60% CER
real-data baseline before you touch synthetic images at all.

Usage:
    python build_qwen_manifest.py \
        --images_dir /content/DiffusionRec_Research/Urdu_Word_Dataset/val/images \
        --gt_dir     /content/DiffusionRec_Research/Urdu_Word_Dataset/val/gt_txt \
        --out        /content/calibration_real.csv \
        --n 50 \
        --seed 42
"""

import argparse
import csv
import random
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_ground_truth(txt_path: Path) -> str:
    """
    Reads a per-image ground-truth .txt file. Handles the common case where
    the file has trailing newlines/whitespace, and is explicit about encoding
    since this is Urdu text (must be read as UTF-8, not a locale default).
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text.strip()


def build_manifest(images_dir: str, gt_dir: str, out_csv: str, n: int, seed: int):
    images_dir = Path(images_dir)
    gt_dir = Path(gt_dir)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"gt_dir does not exist: {gt_dir}")

    all_images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not all_images:
        raise RuntimeError(f"No image files found in {images_dir} (looked for {IMAGE_EXTS})")

    print(f"Found {len(all_images)} candidate images in {images_dir}")

    # Pair each image with its matching ground-truth file by filename stem.
    # e.g. images/00042.png  <->  gt_txt/00042.txt
    paired = []
    missing_gt = []
    empty_gt = []

    for img_path in all_images:
        gt_path = gt_dir / f"{img_path.stem}.txt"
        if not gt_path.is_file():
            missing_gt.append(img_path.name)
            continue
        gt_text = load_ground_truth(gt_path)
        if not gt_text:
            empty_gt.append(img_path.name)
            continue
        paired.append((img_path, gt_text))

    print(f"Paired successfully: {len(paired)}")
    if missing_gt:
        print(f"WARNING: {len(missing_gt)} images had no matching .txt in {gt_dir} "
              f"(first few: {missing_gt[:5]})")
    if empty_gt:
        print(f"WARNING: {len(empty_gt)} ground-truth files were empty after stripping "
              f"(first few: {empty_gt[:5]})")

    if len(paired) == 0:
        raise RuntimeError(
            "No valid (image, ground_truth) pairs found. Check that gt_dir filenames "
            "match images_dir filenames (same stem, .txt extension), and that "
            "gt_preprocessing.py actually ran on this split."
        )

    if n > len(paired):
        print(f"Requested n={n} but only {len(paired)} valid pairs exist — using all of them.")
        n = len(paired)

    rng = random.Random(seed)
    sample = rng.sample(paired, n)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth"])
        for img_path, gt_text in sample:
            # Absolute path so the manifest works regardless of what directory
            # qwen_eval.py is later run from (important in Colab, where the
            # working directory can shift between cells/sessions).
            writer.writerow([str(img_path.resolve()), gt_text])

    print(f"\nWrote {len(sample)} samples to {out_path.resolve()}")
    print("Sample preview:")
    for img_path, gt_text in sample[:5]:
        print(f"  {img_path.name}  ->  {gt_text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Folder of raw word images")
    ap.add_argument("--gt_dir", required=True, help="Folder of per-image ground-truth .txt files")
    ap.add_argument("--out", default="calibration_manifest.csv", help="Output CSV path")
    ap.add_argument("--n", type=int, default=50, help="Number of samples to draw")
    ap.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    args = ap.parse_args()

    build_manifest(args.images_dir, args.gt_dir, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()