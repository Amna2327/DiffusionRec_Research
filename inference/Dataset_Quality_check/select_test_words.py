"""
Step 1 of Test 1: select ~50-100 words from val for the canonical-distance
analysis, and record one real image path per word.

Output columns standardized to (image_path, ground_truth) to match the
format qwen_eval.py and generate_word_samples.py already use, so all three
scripts' CSVs are interchangeable without column-mapping flags.

Auto-resolves the repo root (walks up from this file's location), so you
can run it from anywhere without typing full paths — override with
--gt_folder / --images_folder if your layout ever differs.

Usage (now works with no args, from this script's actual location):
  python select_test_words.py --n 75 --seed 42
"""

import os
import csv
import random
import argparse

IMG_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

DEFAULT_GT_FOLDER = os.path.join(REPO_ROOT, "Urdu_Word_Dataset", "val", "gt_txt")
DEFAULT_IMAGES_FOLDER = os.path.join(REPO_ROOT, "Urdu_Word_Dataset", "val", "images")


def require_dir(path, label):
    if not os.path.isdir(path):
        parent = os.path.dirname(path)
        siblings = os.listdir(parent) if os.path.isdir(parent) else []
        raise FileNotFoundError(
            f"{label} not found: {path}\n"
            f"Contents of parent folder ({parent}): {siblings}\n"
            f"-> Check the exact folder name above and pass --{'gt_folder' if 'gt' in label.lower() else 'images_folder'} explicitly if it differs."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_folder", default=DEFAULT_GT_FOLDER)
    ap.add_argument("--images_folder", default=DEFAULT_IMAGES_FOLDER)
    ap.add_argument("--out_csv", default="test1_words.csv")
    ap.add_argument("--out_wordlist", default="test1_wordlist.txt")
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Resolved repo root: {REPO_ROOT}")
    print(f"Using gt_folder:     {args.gt_folder}")
    print(f"Using images_folder: {args.images_folder}")

    require_dir(args.gt_folder, "gt_folder")
    require_dir(args.images_folder, "images_folder")

    entries = []  # (word, real_image_path)
    for fname in os.listdir(args.gt_folder):
        if not fname.endswith(".txt"):
            continue
        base = os.path.splitext(fname)[0]
        img_path = None
        for ext in IMG_EXTS:
            cand = os.path.join(args.images_folder, base + ext)
            if os.path.exists(cand):
                img_path = cand
                break
        if img_path is None:
            continue
        with open(os.path.join(args.gt_folder, fname), "r", encoding="utf-8") as f:
            word = f.read().strip()
        if word:
            entries.append((word, img_path))

    print(f"Found {len(entries)} (word, image) pairs")

    if not entries:
        raise RuntimeError(
            "Zero pairs found. Either gt_folder has no .txt files, or none of them "
            "have a matching image with the same base filename in images_folder — "
            "check filename conventions (extension, casing, leading zeros) match between the two folders."
        )

    seen = {}
    for word, path in entries:
        if word not in seen:
            seen[word] = path
    unique_words = list(seen.items())
    print(f"{len(unique_words)} unique words available")

    rng = random.Random(args.seed)
    n = min(args.n, len(unique_words))
    selected = rng.sample(unique_words, n)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth"])
        for word, path in selected:
            writer.writerow([path, word])

    with open(args.out_wordlist, "w", encoding="utf-8") as f:
        for word, _ in selected:
            f.write(word + "\n")

    print(f"Selected {n} words (seed={args.seed})")
    print(f"Wrote {args.out_csv} (image_path, ground_truth)")
    print(f"Wrote {args.out_wordlist} (plain word list, for generation)")


if __name__ == "__main__":
    main()