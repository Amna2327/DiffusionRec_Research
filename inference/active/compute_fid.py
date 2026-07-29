"""
FID between your generated word images and real handwriting images.

v2: fixes an aspect-ratio distortion bug in the first version. Real word
images vary in width (different word lengths); squashing everything to a
square distorts each real image by a *different* amount depending on its
original aspect ratio, while every generated image (fixed 64x256) gets
squashed identically. That asymmetric distortion can inflate FID on its
own, independent of actual generation quality. Fix: aspect-preserving
resize + center-pad onto a white canvas, matching the exact approach
WordGenerationDataset._process_image uses during training, so both sets
get comparable treatment.

pip install pytorch-fid

Usage:
  python compute_fid.py \
    --real_dir /content/DiffusionRec_Research/Urdu_Word_Dataset/val/images \
    --generated_dir generated_samples \
    --batch_size 50
"""

import os
import argparse
import tempfile

from PIL import Image
from tqdm import tqdm
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths


IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def image_resize_preserve_aspect(img, height):
    w, h = img.size
    if h == 0:
        return img
    ratio = height / h
    new_w = max(1, int(w * ratio))
    return img.resize((new_w, height), Image.LANCZOS)


def center_pad_canvas(img, canvas_size, border_value=255):
    """canvas_size = (height, width). Mirrors centered_PIL in
    WordGenerationDataset -- same treatment real training data gets."""
    target_h, target_w = canvas_size
    if img.mode != "RGB":
        img = img.convert("RGB")
    canvas = Image.new("RGB", (target_w, target_h), (border_value,) * 3)
    img_w, img_h = img.size
    paste_x = max(0, (target_w - img_w) // 2)
    paste_y = max(0, (target_h - img_h) // 2)
    canvas.paste(img.crop((0, 0, min(img_w, target_w), min(img_h, target_h))), (paste_x, paste_y))
    return canvas


def preprocess_to_fixed_canvas(src_dir, dst_dir, canvas_size=(64, 256)):
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.endswith(IMG_EXTS)]
    print(f"Preprocessing {len(files)} images from {src_dir} -> {dst_dir} (aspect-preserving, canvas={canvas_size})")
    for fname in tqdm(files):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        img = image_resize_preserve_aspect(img, height=canvas_size[0] - 4)  # small margin, matches training's -4
        img = center_pad_canvas(img, canvas_size)
        img.save(os.path.join(dst_dir, fname))
    return dst_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--dims", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--canvas_h", type=int, default=64)
    ap.add_argument("--canvas_w", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    canvas = (args.canvas_h, args.canvas_w)

    with tempfile.TemporaryDirectory() as tmp:
        real_resized = preprocess_to_fixed_canvas(args.real_dir, os.path.join(tmp, "real"), canvas)
        gen_resized = preprocess_to_fixed_canvas(args.generated_dir, os.path.join(tmp, "gen"), canvas)

        fid_value = calculate_fid_given_paths(
            [real_resized, gen_resized],
            batch_size=args.batch_size,
            device=device,
            dims=args.dims,
            num_workers=args.num_workers,
        )

    print("\n" + "=" * 50)
    print(f"FID: {fid_value:.2f}")
    print("=" * 50)
    print("\nFor reference, the paper reports (Table 1, word-level):")
    print("  DiffusionRec (theirs): FID 19.35")
    print("  OneDM baseline:        FID 99.6")
    print("  DiffBrush baseline:    FID 111")


if __name__ == "__main__":
    main()