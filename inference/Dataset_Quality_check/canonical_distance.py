"""
Step 3/5 of Test 1: for each (word, image) pair in a manifest, render the
word's canonical PRINTED form and measure how far the actual image sits
from it in Inception feature space. Run this once on the real manifest and
once on the generated manifest -- same script, reusable, so X_real and X_gen
are computed identically (apples to apples).

IMPORTANT: sanity-check the canonical rendering before trusting anything
downstream. Urdu/Arabic needs contextual letter shaping (initial/medial/
final/isolated forms) that plain PIL text rendering does NOT do on its own
UNLESS Pillow is built with Raqm support (HarfBuzz + FriBidi under the
hood), which handles shaping and bidi reordering automatically. This
script relies on that -- confirmed present via PIL.features.check("raqm").
Run with --preview_word first and LOOK at the output image before running
the full manifest.

Usage (sanity check first):
  python canonical_distance.py --preview_word "کون" --font_path urdu_font.ttf

Usage (real run — remember --word_col/--image_col if your manifest's
column names don't match the defaults):
  python canonical_distance.py \
    --manifest test1_words.csv \
    --word_col ground_truth \
    --font_path urdu_font.ttf \
    --out_csv test1_real_distances.csv

  python canonical_distance.py \
    --manifest generated_samples_test1/manifest.csv \
    --word_col ground_truth \
    --font_path urdu_font.ttf \
    --out_csv test1_gen_distances.csv
"""

import os
import csv
import argparse

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pytorch_fid.inception import InceptionV3

def render_canonical(word: str, font_path: str, base_font_size: int, canvas_size=(64, 256)):
    canvas_h, canvas_w = canvas_size
    font_size = base_font_size
    font = ImageFont.truetype(font_path, font_size)
    dummy = Image.new("RGB", (canvas_w, canvas_h))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), word, font=font)
    text_h = bbox[3] - bbox[1]
    text_w = bbox[2] - bbox[0]

    margin = 8
    while (text_h > canvas_h - margin or text_w > canvas_w - margin) and font_size > 8:
        font_size -= 2
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), word, font=font)
        text_h = bbox[3] - bbox[1]
        text_w = bbox[2] - bbox[0]

    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = max(0, (canvas_w - text_w) // 2 - bbox[0])
    y = max(0, (canvas_h - text_h) // 2 - bbox[1])
    draw.text((x, y), word, fill=(0, 0, 0), font=font)
    return img

def load_inception(device):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device).eval()
    return model


@torch.no_grad()
def extract_feature(img: Image.Image, model, device):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    feat = model(tensor)[0]
    feat = feat.squeeze(-1).squeeze(-1).squeeze(0).cpu().numpy()
    return feat


def cosine_distance(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return 1.0 - float(np.dot(a_norm, b_norm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None, help="CSV with a word column and an image path column")
    ap.add_argument("--word_col", default="word")
    ap.add_argument("--image_col", default="image_path")
    ap.add_argument("--font_path", default="urdu_font.ttf")
    ap.add_argument("--font_size", type=int, default=28)
    ap.add_argument("--canvas_h", type=int, default=64)
    ap.add_argument("--canvas_w", type=int, default=256)
    ap.add_argument("--out_csv", default="distances.csv")
    ap.add_argument("--preview_word", default=None,
                     help="Render just this one word and save canonical_preview.png -- "
                          "SANITY CHECK THIS before running the full manifest")
    args = ap.parse_args()

    canvas = (args.canvas_h, args.canvas_w)

    if args.preview_word:
        img = render_canonical(args.preview_word, args.font_path, args.font_size, canvas)
        img.save("canonical_preview.png")
        print(f"Saved canonical_preview.png for word: {args.preview_word}")
        print("LOOK AT THIS IMAGE before running the full manifest. It should show")
        print("properly joined, connected Nastaliq-style letterforms -- not floating,")
        print("disconnected individual characters. If it looks wrong, do not proceed.")
        return

    if not args.manifest:
        raise ValueError("--manifest required unless using --preview_word")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Inception-V3 feature extractor on {device}...")
    model = load_inception(device)

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"Processing {len(rows)} rows from {args.manifest}")

    results = []
    for row in rows:
        word = row[args.word_col]
        img_path = row[args.image_col]
        try:
            canonical_img = render_canonical(word, args.font_path, args.font_size, canvas)
            actual_img = Image.open(img_path)

            canonical_feat = extract_feature(canonical_img, model, device)
            actual_feat = extract_feature(actual_img, model, device)
            dist = cosine_distance(canonical_feat, actual_feat)

            results.append({"word": word, "image_path": img_path, "canonical_distance": dist})
        except Exception as e:
            print(f"  error on word={word!r} ({img_path}): {e}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "image_path", "canonical_distance"])
        writer.writeheader()
        writer.writerows(results)

    dists = [r["canonical_distance"] for r in results]
    print(f"\nWrote {len(results)} rows to {args.out_csv}")
    print(f"canonical_distance: mean={np.mean(dists):.4f}, std={np.std(dists):.4f}, "
          f"min={np.min(dists):.4f}, max={np.max(dists):.4f}")

if __name__ == "__main__":
    main()