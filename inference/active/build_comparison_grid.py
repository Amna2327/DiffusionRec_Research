"""
Pull matching real/generated word image pairs (same ground-truth word) and
lay them out side by side for a quick visual sanity check -- exactly the
"just look at the images" step before trusting or distrusting a FID number.

Usage:
  python build_comparison_grid.py \
    --generated_manifest generated_samples/manifest.csv \
    --real_gt_folder /content/DiffusionRec_Research/Urdu_Word_Dataset/val/gt \
    --real_images_folder /content/DiffusionRec_Research/Urdu_Word_Dataset/val/images \
    --out_dir comparison_grid \
    --n_pairs 10 \
    --seed 42
"""

import os
import csv
import random
import argparse

from PIL import Image, ImageDraw, ImageFont


IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
ROW_H = 140          # height allotted per image row (image + label)
IMG_H = 100           # height each individual image gets, aspect-ratio preserved
PANEL_W = 420         # width of each (real / generated) panel
LABEL_H = 40
FONT_PATH_CANDIDATES = [
    "urdu_font.ttf",
    "tokenizer_assets/urdu_font.ttf",
    "../tokenizer_assets/urdu_font.ttf",
]


def find_font(size=22):
    for p in FONT_PATH_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    print("Warning: urdu_font.ttf not found in expected locations -- word labels "
          "may not render Urdu glyphs correctly. Pass --font_path explicitly if needed.")
    return ImageFont.load_default()


def build_real_word_index(gt_folder, images_folder):
    """word -> list of real image paths with that transcription"""
    index = {}
    txt_files = [f for f in os.listdir(gt_folder) if f.endswith(".txt")]
    for txt_file in txt_files:
        base = os.path.splitext(txt_file)[0]
        img_path = None
        for ext in IMG_EXTS:
            candidate = os.path.join(images_folder, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            continue
        with open(os.path.join(gt_folder, txt_file), "r", encoding="utf-8") as f:
            word = f.read().strip()
        if word:
            index.setdefault(word, []).append(img_path)
    return index


def load_generated_manifest(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["image_path"], row["ground_truth"]))
    return rows


def fit_image(img_path, target_h):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if h == 0:
        return img
    ratio = target_h / h
    new_w = max(1, int(w * ratio))
    return img.resize((new_w, target_h), Image.LANCZOS)


def make_panel(img_path, label_text, font):
    panel = Image.new("RGB", (PANEL_W, ROW_H), (255, 255, 255))
    img = fit_image(img_path, IMG_H)
    img = img.crop((0, 0, min(img.width, PANEL_W - 10), IMG_H))  # avoid overflow on long words
    panel.paste(img, (5, 5))
    draw = ImageDraw.Draw(panel)
    draw.text((5, IMG_H + 10), label_text, fill=(0, 0, 0), font=font)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated_manifest", required=True)
    ap.add_argument("--real_gt_folder", required=True)
    ap.add_argument("--real_images_folder", required=True)
    ap.add_argument("--out_dir", default="comparison_grid")
    ap.add_argument("--n_pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--font_path", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    font = ImageFont.truetype(args.font_path, 22) if args.font_path else find_font()

    print("Indexing real images by ground-truth word...")
    real_index = build_real_word_index(args.real_gt_folder, args.real_images_folder)
    print(f"Indexed {len(real_index)} unique real words")

    generated = load_generated_manifest(args.generated_manifest)
    print(f"Loaded {len(generated)} generated samples")

    matched = [(gen_path, word) for gen_path, word in generated if word in real_index]
    print(f"{len(matched)}/{len(generated)} generated words have a matching real image with the same ground truth")

    if not matched:
        print("No matches found -- real and generated vocabularies don't overlap. "
              "Falling back to unmatched random pairs so you can still eyeball style/quality "
              "(NOTE: words will differ between columns in this fallback).")
        rng = random.Random(args.seed)
        real_words = list(real_index.keys())
        sample_gen = rng.sample(generated, min(args.n_pairs, len(generated)))
        pairs = []
        for gen_path, gen_word in sample_gen:
            real_word = rng.choice(real_words)
            real_path = rng.choice(real_index[real_word])
            pairs.append((real_path, real_word, gen_path, gen_word))
    else:
        rng = random.Random(args.seed)
        sample = rng.sample(matched, min(args.n_pairs, len(matched)))
        pairs = []
        for gen_path, word in sample:
            real_path = rng.choice(real_index[word])
            pairs.append((real_path, word, gen_path, word))

    # Build contact sheet: header row + one row per pair, two columns (real | generated)
    n = len(pairs)
    sheet_w = PANEL_W * 2 + 20
    sheet_h = ROW_H * (n + 1) + 20
    sheet = Image.new("RGB", (sheet_w, sheet_h), (240, 240, 240))
    draw = ImageDraw.Draw(sheet)
    header_font = find_font(26) if args.font_path is None else ImageFont.truetype(args.font_path, 26)
    draw.text((10, 5), "REAL", fill=(0, 0, 0), font=header_font)
    draw.text((PANEL_W + 15, 5), "GENERATED", fill=(0, 0, 0), font=header_font)

    for i, (real_path, real_word, gen_path, gen_word) in enumerate(pairs):
        y = ROW_H * (i + 1) + 20
        real_panel = make_panel(real_path, real_word, font)
        gen_panel = make_panel(gen_path, gen_word, font)
        sheet.paste(real_panel, (5, y))
        sheet.paste(gen_panel, (PANEL_W + 10, y))

    out_path = os.path.join(args.out_dir, "comparison_sheet.png")
    sheet.save(out_path)
    print(f"\nSaved contact sheet: {out_path}")
    print(f"({n} pairs, {'matched by shared ground-truth word' if matched else 'UNMATCHED fallback -- words differ'})")


if __name__ == "__main__":
    main()