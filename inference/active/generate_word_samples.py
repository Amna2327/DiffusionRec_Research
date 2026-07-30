"""
Generate a batch of DiffusionRec word images from the trained EMA checkpoint,
sampling target words from the real test-set ground truth, and write a
manifest CSV in the same (image_path, ground_truth) format qwen_ocr_eval.py
already expects.

Run from the same directory as unet.py (needs it importable).

Usage:
  python generate_word_samples.py \
    --ckpt_dir /content/DiffusionRec_Research/word_level_model/models \
    --test_gt_folder /content/DiffusionRec_Research/Urdu_Word_Dataset/test/gt_txt \
    --out_dir generated_samples \
    --n_words 200 \
    --seed 42 \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5
"""

import os
import sys
sys.path.append(os.path.abspath("/content/DiffusionRec_Research/models"))
import csv
import random
import argparse
from types import SimpleNamespace

import torch
import torchvision
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

from unet import UNetModel


def strip_module_prefix(state_dict):
    """Checkpoints were saved from a DataParallel-wrapped UNet whose
    text_encoder submodule was ALSO separately DataParallel-wrapped before
    being passed in (confirmed by the nested 'text_encoder.module.xxx' keys
    in the checkpoint). So there are two layers of 'module.' to remove, not
    one, and not both at a fixed position. Strip any 'module' path segment
    wherever it appears."""
    new_sd = {}
    for k, v in state_dict.items():
        parts = [p for p in k.split(".") if p != "module"]
        new_sd[".".join(parts)] = v
    return new_sd


def sample_target_words(gt_folder: str, n: int, seed: int):
    words = []
    for fname in os.listdir(gt_folder):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(gt_folder, fname), "r", encoding="utf-8") as f:
            w = f.read().strip()
        if w:
            words.append(w)
    print(f"Found {len(words)} candidate words in {gt_folder}")
    rng = random.Random(seed)
    if n >= len(words):
        print(f"Requested n={n} >= available {len(words)}; using all of them.")
        return words
    return rng.sample(words, n)


@torch.no_grad()
def generate_batch(unet, vae, tokenizer, text_encoder, words, args, batch_size=8):
    """Text-conditioned DDIM sampling, mirroring Diffusion.sampling() in the
    training script but standalone so we don't need to import the whole
    training module (and its recognizer-loading side effects) just to sample."""
    beta = torch.linspace(1e-4, 2e-2, 1000).to(args.device)  # unused directly; scheduler handles noise math
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    ddim.set_timesteps(50)

    results = []  # list of (word, PIL image)
    for i in range(0, len(words), batch_size):
        batch_words = words[i:i + batch_size]
        n = len(batch_words)

        text_features = tokenizer(batch_words, padding="max_length", truncation=True,
                                   return_tensors="pt", max_length=200)
        text_features = {k: v.to(args.device) for k, v in text_features.items()}

        # Confirmed from WordGenerationDataset: word-level style_id is always 0
        # (self.wclasses = 1), so this matches training exactly rather than
        # guessing at a "no conditioning" path that doesn't actually exist.
        labels = torch.zeros(n, dtype=torch.long, device=args.device)

        x = torch.randn((n, 4, args.img_size[0] // 8, args.img_size[1] // 8), device=args.device)
        ddim.set_timesteps(50)

        for t in ddim.timesteps:
            t_batch = (torch.ones(n) * t.item()).long().to(args.device)
            noise_pred = unet(x, t_batch, text_features, labels, style_extractor=None)
            x = ddim.step(noise_pred, t, x).prev_sample

        latents = x / 0.18215
        images = vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)

        for j, word in enumerate(batch_words):
            grid = torchvision.utils.make_grid(images[j].unsqueeze(0))
            ndarr = grid.permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray((ndarr * 255).astype("uint8"))
            results.append((word, im))

        print(f"Generated {i + n}/{len(words)}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True, help="Folder containing ema_ckpt.pt")
    ap.add_argument("--test_gt_folder", required=True)
    ap.add_argument("--out_dir", default="generated_samples")
    ap.add_argument("--n_words", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--stable_dif_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = ap.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    args = SimpleNamespace(
        device=args_cli.device,
        img_size=(64, 256),
        stable_dif_path=args_cli.stable_dif_path,
    )

    print("Loading CANINE text encoder...")
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c").to(args.device).eval()

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(args.device).eval()
    vae.requires_grad_(False)

    print("Building UNet (num_classes=1, matching confirmed word-level dummy style dim)...")
    unet_args = SimpleNamespace(interpolation=False, mix_rate=None)
    unet = UNetModel(
        image_size=args.img_size, in_channels=4, model_channels=320, out_channels=4,
        num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
        num_heads=4, num_classes=1, context_dim=320, vocab_size=80,
        text_encoder=text_encoder, args=unet_args,
    ).to(args.device).eval()

    ckpt_path = os.path.join(args_cli.ckpt_dir, "ema_ckpt.pt")
    print(f"Loading EMA weights from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=args.device)
    unet.load_state_dict(strip_module_prefix(state_dict))
    unet.requires_grad_(False)

    words = sample_target_words(args_cli.test_gt_folder, args_cli.n_words, args_cli.seed)
    print(f"Sampled {len(words)} target words (seed={args_cli.seed})")

    results = generate_batch(unet, vae, tokenizer, text_encoder, words, args, args_cli.batch_size)

    manifest_path = os.path.join(args_cli.out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "ground_truth"])
        for idx, (word, img) in enumerate(results):
            img_path = os.path.join(args_cli.out_dir, f"{idx:05d}.png")
            img.save(img_path)
            writer.writerow([img_path, word])

    print(f"\nDone. {len(results)} images written to {args_cli.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"\nNext: python qwen_ocr_eval.py --checkpoint <qwen_dir> --manifest {manifest_path}")


if __name__ == "__main__":
    main()