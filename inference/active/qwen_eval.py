"""
Qwen3-VL-4B-Instruct OCR inference + Recognition Accuracy eval harness
for DiffusionRec word-level reproduction.

This exists because the original fine-tuning/eval code for Qwen3-VL is lost;
you have the fine-tuned WEIGHTS, so this rebuilds a minimal, honest inference
+ scoring path around them. Read the CALIBRATION note at the bottom before
trusting numbers from this at scale.

pip install -U transformers accelerate pillow
(qwen_vl_utils is optional here since we're doing single local images;
 included as a fallback import in case you already have it installed)
"""

import os
import csv
import json
import argparse
import unicodedata
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ---------------------------------------------------------------------------
# CONFIG — fill these in / pass via CLI. Nothing here is guessed at runtime;
# everything you need to check is called out explicitly.
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "Read the Urdu handwritten word in this image. Reply with only the transcription, nothing else."

# If your fine-tuned checkpoint was trained with a specific instruction phrasing
# (you won't know this for certain since the fine-tuning code is lost), try a
# couple of alternates during calibration:
PROMPT_VARIANTS = {
    "plain": DEFAULT_PROMPT,
    "ocr_style": "Transcribe the handwritten Urdu text in this image.",
    "minimal": "What word is written in this image?",
}


def load_model(checkpoint_path: str, device: str = "auto"):
    """
    checkpoint_path: local dir with your fine-tuned safetensors + config.json,
    OR a HF hub id if you haven't got local weights.
    """
    print(f"Loading Qwen3-VL from: {checkpoint_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        dtype="auto",
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model.eval()
    return model, processor


@torch.no_grad()
def run_ocr(model, processor, image: Image.Image, prompt: str, max_new_tokens: int = 64) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy — deterministic, matches an accuracy-eval setting
    )
    # Slice off the prompt tokens so we only decode the newly generated part
    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    result = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return result


def normalize_urdu(s: str) -> str:
    """
    Minimal normalization before comparing predictions to ground truth.
    NFC normalization + strip whitespace. Deliberately NOT stripping diacritics
    or doing anything more aggressive — over-normalizing can silently inflate
    accuracy. Extend only if you've checked it doesn't hide real errors.
    """
    s = unicodedata.normalize("NFC", s)
    return s.strip()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def evaluate(manifest_csv: str, model, processor, prompt: str, out_csv: str, max_samples: int = None):
    """
    manifest_csv: CSV with columns `image_path,ground_truth`
    Writes per-sample predictions to out_csv, prints aggregate metrics.

    NOTE on metric definition: the paper's word-level "Rec. Acc." is not
    explicitly defined in the text I have (only sentence-level is spelled out
    as 1-CER). This script reports BOTH exact-match accuracy and (1-CER) so
    you can see which one, if either, lines up with the paper's 45.2%/74.4%
    once you calibrate below. Don't assume which one is "correct" until you
    check.
    """
    rows = []
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if max_samples:
        rows = rows[:max_samples]

    results = []
    n_exact = 0
    total_cer_num = 0
    total_cer_den = 0

    for i, row in enumerate(rows):
        img_path = row["image_path"]
        gt = normalize_urdu(row["ground_truth"])
        try:
            img = Image.open(img_path).convert("RGB")
            pred_raw = run_ocr(model, processor, img, prompt)
            pred = normalize_urdu(pred_raw)
        except Exception as e:
            print(f"[{i}] ERROR on {img_path}: {e}")
            pred_raw, pred = "", ""

        exact = int(pred == gt)
        dist = levenshtein(pred, gt)
        n_exact += exact
        total_cer_num += dist
        total_cer_den += max(len(gt), 1)

        results.append({
            "image_path": img_path,
            "ground_truth": gt,
            "prediction_raw": pred_raw,
            "prediction_normalized": pred,
            "exact_match": exact,
            "char_edit_distance": dist,
        })

        if i % 25 == 0:
            print(f"[{i}/{len(rows)}] gt={gt!r} pred={pred!r} exact={exact}")

    n = len(rows)
    exact_acc = n_exact / n if n else 0.0
    cer = total_cer_num / total_cer_den if total_cer_den else 0.0
    rec_acc_via_cer = 1 - cer

    print("\n" + "=" * 60)
    print(f"Samples evaluated:        {n}")
    print(f"Exact-match accuracy:     {exact_acc*100:.2f}%")
    print(f"CER:                      {cer*100:.2f}%")
    print(f"Rec. Acc. (1 - CER):      {rec_acc_via_cer*100:.2f}%")
    print("=" * 60)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else [])
        writer.writeheader()
        writer.writerows(results)
    print(f"Per-sample predictions written to {out_csv}")

    summary = {
        "n_samples": n,
        "exact_match_accuracy": exact_acc,
        "cer": cer,
        "rec_acc_1_minus_cer": rec_acc_via_cer,
        "prompt_used": prompt,
    }
    with open(out_csv.replace(".csv", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to your fine-tuned Qwen3-VL weights (local dir or HF hub id)")
    ap.add_argument("--manifest", required=True, help="CSV with columns: image_path,ground_truth")
    ap.add_argument("--out", default="qwen_ocr_predictions.csv")
    ap.add_argument("--prompt_variant", default="plain", choices=list(PROMPT_VARIANTS.keys()))
    ap.add_argument("--max_samples", type=int, default=None, help="Limit for calibration runs, e.g. 20-50")
    args = ap.parse_args()

    model, processor = load_model(args.checkpoint)
    prompt = PROMPT_VARIANTS[args.prompt_variant]
    evaluate(args.manifest, model, processor, prompt, args.out, args.max_samples)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# CALIBRATION — do this before trusting any number this script produces:
#
# 1. Build a small manifest (~30-50 samples) of REAL handwritten test images
#    from IIIT-INDIC-HW-WORDS-Urdu with their ground truth. Run this script
#    against it with --max_samples 50.
# 2. Compare against the paper's real-data baseline: 74.40% Rec. Acc. / 25.60%
#    CER (Section 5.1). If your numbers are close, your prompt + preprocessing
#    are reasonable stand-ins for whatever the original eval harness did.
#    If they're wildly off (e.g. near 0%, or garbled output), the prompt
#    template almost certainly doesn't match what the model was fine-tuned to
#    expect — try the other PROMPT_VARIANTS, and check model.generation_config
#    / chat_template.json in the checkpoint dir for any hints the authors left.
# 3. Only once step 2 is in a believable range should you run this against
#    your own DiffusionRec-generated word images and treat the resulting
#    number as comparable to the paper's 45.2%.
# 4. Exact-match vs (1-CER): report both until you have evidence for which
#    one the paper actually used at word level — don't silently pick one.
# ---------------------------------------------------------------------------