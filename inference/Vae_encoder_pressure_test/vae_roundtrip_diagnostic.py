"""
vae_roundtrip_diagnostic.py

Isolates whether SD1.5's VAE bottleneck itself is destroying Urdu
handwriting legibility, independent of the UNet/diffusion process or the
recognition-loss question entirely.

Takes real images (never touched by diffusion generation), encodes them
through the VAE, immediately decodes back -- no noise, no UNet, no
denoising steps -- and saves the round-tripped result. Feed the output
manifest into qwen_eval.py and compare against the already-known real-image
baseline (65.56% Rec. Acc. from earlier calibration).

If round-tripped images score close to the original 65.56%: VAE is fine,
the bottleneck is elsewhere (UNet training / recognition-loss gradient
path). If it drops sharply: the VAE itself is destroying handwriting-
critical detail before anything else even runs, and no UNet-side fix would
solve that.

Usage:
  python vae_roundtrip_diagnostic.py \
    --manifest test1_words.csv \
    --word_col ground_truth \
    --image_col image_path \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --out_dir vae_roundtrip_samples \
    --out_manifest vae_roundtrip_manifest.csv
"""

import os
import csv
import argparse

import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL


IMG_WIDTH = 256
IMG_HEIGHT = 64


def require_file(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_and_preprocess(img_path: str, transform) -> torch.Tensor:
    """Resize to the fixed training canvas size and apply the same
    ToTensor + Normalize(0.5, 0.5, 0.5) transform used everywhere else in
    this codebase, so the VAE sees exactly what it sees during real
    training -- not a differently-preprocessed stand-in."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
    return transform(img)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Inverse of the (x/2 + 0.5).clamp(0,1) denormalization used
    throughout the training/sampling code, -> uint8 PIL image."""
    img = (img_tensor / 2 + 0.5).clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV of real images, e.g. test1_words.csv")
    ap.add_argument("--word_col", default="ground_truth")
    ap.add_argument("--image_col", default="image_path")
    ap.add_argument("--stable_dif_path", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--out_dir", default="vae_roundtrip_samples")
    ap.add_argument("--out_manifest", default="vae_roundtrip_manifest.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    require_file(args.manifest, "manifest")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading VAE from {args.stable_dif_path} ...")
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(args.device).eval()
    vae.requires_grad_(False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if args.max_samples:
        rows = rows[: args.max_samples]
    print(f"Processing {len(rows)} real images from {args.manifest}")

    out_rows = []
    n_errors = 0
    for idx, row in enumerate(rows):
        word = row[args.word_col]
        img_path = row[args.image_col]
        try:
            x = load_and_preprocess(img_path, transform).unsqueeze(0).to(args.device)

            latents = vae.encode(x).latent_dist.sample() * 0.18215
            recon = vae.decode(latents / 0.18215).sample

            out_img = tensor_to_pil(recon.squeeze(0))
            out_path = os.path.join(args.out_dir, f"{idx:05d}.png")
            out_img.save(out_path)
            out_rows.append({"image_path": out_path, "ground_truth": word})

        except Exception as e:
            n_errors += 1
            print(f"  error on word={word!r} ({img_path}): {e}")

        if idx % 10 == 0:
            print(f"[{idx}/{len(rows)}]")

    with open(args.out_manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "ground_truth"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone. {len(out_rows)} round-tripped images written to {args.out_dir}")
    print(f"Errors: {n_errors}")
    print(f"Manifest: {args.out_manifest}")
    print(f"\nNext: python qwen_eval.py --checkpoint <qwen_dir> --manifest {args.out_manifest} --out vae_roundtrip_qwen_predictions.csv")


if __name__ == "__main__":
    main()