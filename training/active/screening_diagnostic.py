"""
screening_diagnostic.py

Cheap, fast diagnostic to test whether the recognition-loss gradient fix
actually moves recognition accuracy, before committing to full 71k-word
training runs.

Design (see DiffusionRec_Recognition_Gap diagnostic doc):
  - Train on a SMALL rotating subset of the training pool (not the full
    71,207 words) -- subset is resampled every --rotate_every epochs so no
    single image gets memorized via repeated exposure across the whole run.
  - Recognition-loss curriculum is cranked aggressively (--rec_weight_start,
    --rec_weight_max, --rec_curriculum_epochs, --rec_start_epoch=0) on
    purpose -- this is a diagnostic config, not a realistic training recipe.
  - Held-out eval words are drawn from the disjoint val set (never touched by
    the original full training run), not carved out of train -- carving from
    train only excludes words from THIS diagnostic's subset rotation, but the
    base checkpoint already saw every train word during full training, so
    that wouldn't measure real generalization.
  - --control lets you flip back to the OLD (broken/detached) gradient path
    so you can run a same-subset, same-curriculum control and isolate
    whether it's really the fix doing the work, not just aggressive loss
    weighting perturbing training.

Paths below match the repo layout:
  training/active/train_word_level_with_resume_v2.py  -- main training script
  inference/active/qwen_eval.py                        -- Qwen3-VL OCR eval harness

Run with --qwen_checkpoint pointing at your fine-tuned Qwen3-VL weights to get
real Rec.Acc./CER numbers; without it, eval images + manifests are still saved
to disk each eval pass, just not scored.
"""

import os
import sys
import csv
import json
import random
import argparse
import copy
import gc

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from transformers import CanineModel, CanineTokenizer
from torch.nn import DataParallel

REPO_ROOT = '/content/DiffusionRec_Research'

# models/ and data/ live directly under REPO_ROOT, not under training/active or
# inference/active -- needs to be on the path for `from models.unet import
# UNetModel` etc. to resolve, both here and inside the imported train module.
sys.path.append(REPO_ROOT)

# --- main training script (training/active/train_word_level_with_resume_v2.py) ---
sys.path.append(os.path.join(REPO_ROOT, 'training', 'active'))
TRAIN_MODULE = 'train_word_level_with_resume_v2'
_train_mod = __import__(TRAIN_MODULE)
str2bool = _train_mod.str2bool
setup_logging = _train_mod.setup_logging
save_images = _train_mod.save_images
AvgMeter = _train_mod.AvgMeter
EMA = _train_mod.EMA
Diffusion = _train_mod.Diffusion
load_urdu_recognizer = _train_mod.load_urdu_recognizer
recognize_urdu_batch = _train_mod.recognize_urdu_batch
preprocess_image_for_recognizer_torch = _train_mod.preprocess_image_for_recognizer_torch
load_checkpoint = _train_mod.load_checkpoint
gpu_mem = _train_mod.gpu_mem
from models.unet import UNetModel
from data.utils.word_generation_dataset import WordGenerationDataset
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast as _autocast, GradScaler as _GradScaler

# --- Qwen OCR eval harness (inference/active/qwen_eval.py) ---
sys.path.append(os.path.join(REPO_ROOT, 'inference', 'active'))
import qwen_eval

_to_pil = ToPILImage()


def autocast(enabled=True):
    return _autocast('cuda', enabled=enabled)


def GradScaler(enabled=True):
    return _GradScaler('cuda', enabled=enabled)


# ---------------------------------------------------------------------------
# Held-out eval vocab comes from the val set, which is genuinely disjoint from
# train (never touched by the original full training run) -- unlike carving
# words out of train, which the base checkpoint has already seen regardless.
# No indexing/caching needed here: WordGenerationDataset already builds
# self.data as a plain list of (img_path, transcr, style_id, img_path) tuples
# with zero image I/O, so pulling unique words out of val is near-instant even
# without the fast-path/cache machinery train indexing used to need.
# ---------------------------------------------------------------------------

def build_val_held_out_vocab(val_data, n_words, seed):
    words = sorted({item[1] for item in val_data.data})
    rng = random.Random(seed)
    rng.shuffle(words)
    if n_words > len(words):
        print(f"[WARN] --held_out_words ({n_words}) > val vocab size ({len(words)}); using all {len(words)}")
        n_words = len(words)
    held_out_vocab = words[:n_words]
    print(f"Held-out eval vocab: {len(held_out_vocab)} words drawn from val "
          f"({len(words)} unique words / {len(val_data)} images in val) -- "
          f"genuinely unseen by the checkpoint, not just excluded from this run's subset")
    return held_out_vocab


def sample_training_subset(pool_indices, subset_size, rng):
    size = min(subset_size, len(pool_indices))
    return rng.sample(pool_indices, size)




def run_qwen_eval(images, words, args, out_dir, tag):
    """
    Real integration with inference/active/qwen_eval.py.

    qwen_eval.evaluate() works off a manifest CSV of (image_path, ground_truth)
    pairs read from disk -- it's not built to accept in-memory tensors -- so
    we save each generated sample as a PNG, write a manifest, and call it the
    same way you'd run it on real held-out test images. This means diagnostic
    numbers are produced by the exact same scoring code as your paper-baseline
    calibration runs, which is the whole point of reusing it.
    """
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, tag)
    os.makedirs(img_dir, exist_ok=True)

    # keep the quick visual grid for eyeballing, same as before
    save_images(images, os.path.join(out_dir, f"{tag}_grid.jpg"), args, texts=words)

    manifest_path = os.path.join(out_dir, f"{tag}_manifest.csv")
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'ground_truth'])
        pil_images = [_to_pil(img.detach().cpu().clamp(0, 1)) for img in images]
        for idx, (pil_img, word) in enumerate(zip(pil_images, words)):
            img_path = os.path.join(img_dir, f"{idx:03d}.png")
            pil_img.save(img_path)
            writer.writerow([img_path, word])

    if not args.qwen_checkpoint:
        print(f"  [WARN] --qwen_checkpoint not set -- skipping quantitative score "
              f"for '{tag}'. Images + manifest saved to {img_dir} for manual "
              f"inspection or a later qwen_eval.py run.")
        return None

    # Load Qwen fresh for this eval pass and fully discard it afterward --
    # NOT parked on CPU between passes. Parking a 4B-param model on CPU
    # between evals (what the previous version did) was itself the cause of
    # a later crash: it quietly eats several GB of system RAM (not GPU VRAM)
    # for the epochs in between, and Colab's system RAM is much tighter than
    # its GPU VRAM -- with persistent DataLoader workers also holding memory,
    # the OS's OOM killer ended up killing a worker process. Loading fresh
    # each time costs a few seconds per eval pass but leaves nothing idle
    # in between, on GPU or CPU.
    print(f"  Loading Qwen for eval pass '{tag}'...")
    qwen_model, qwen_processor = qwen_eval.load_model(args.qwen_checkpoint, device=args.qwen_device)
    qwen_prompt = qwen_eval.PROMPT_VARIANTS[args.qwen_prompt_variant]

    out_csv = os.path.join(out_dir, f"{tag}_predictions.csv")
    result = qwen_eval.evaluate(manifest_path, qwen_model, qwen_processor, qwen_prompt, out_csv)

    del qwen_model, qwen_processor
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  [{tag}] exact_match={result['exact_match_accuracy']*100:.2f}% "
          f"CER={result['cer']*100:.2f}% Rec.Acc.(1-CER)={result['rec_acc_1_minus_cer']*100:.2f}%")
    return result


def evaluate_holdout(model, vae, diffusion, noise_scheduler, held_out_vocab, num_classes,
                      args, tokenizer, text_encoder, out_dir, tag, eval_sample_count):
    model.eval()
    sample_words = held_out_vocab[:eval_sample_count]
    n = len(sample_words)
    # BUG FIX: must be % num_classes (matches original script's validate()),
    # not % n -- word-level generation has num_classes=1 (a single dummy
    # style), so label_emb is an Embedding(1, ...). % n produced indices
    # 0..n-1, an out-of-bounds gather into a size-1 embedding table the
    # moment n > 1 -- that's what crashed the CANINE forward pass above.
    labels = torch.arange(n).long().to(args.device) % num_classes

    images = diffusion.sampling(
        model, vae, n=n, x_text=sample_words, labels=labels, args=args,
        style_extractor=None, noise_scheduler=noise_scheduler,
        transform=None, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder
    )
    result = run_qwen_eval(images, sample_words, args, out_dir, tag)
    model.train()
    return result


def diagnostic_train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss,
                      train_data, pool_indices, held_out_vocab, num_classes,
                      noise_scheduler, args, tokenizer, text_encoder, scaler, results_log):
    model.train()
    rng = random.Random(args.seed)
    loader = None
    current_subset_words = None

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # --- rotate subset every N epochs ---
        if epoch % args.rotate_every == 0:
            subset_indices = sample_training_subset(pool_indices, args.subset_size, rng)
            loader = DataLoader(
                Subset(train_data, subset_indices),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers,
                persistent_workers=args.num_workers > 0,
                pin_memory=torch.cuda.is_available()
            )
            current_subset_words = sorted({train_data[i][1] for i in subset_indices})
            print(f"\n[Epoch {epoch}] Rotated subset: {len(subset_indices)} images, "
                  f"{len(current_subset_words)} unique words")
            results_log['subset_rotations'].append({'epoch': epoch, 'n_words': len(current_subset_words)})

        # aggressive curriculum, start_epoch defaults to 0
        current_rec_weight = min(
            args.rec_weight_start + (args.rec_weight_max - args.rec_weight_start) * (epoch / max(args.rec_curriculum_epochs, 1)),
            args.rec_weight_max
        )
        print(f"Epoch {epoch} | rec_weight={current_rec_weight:.4f} | control={args.control}")

        pbar = tqdm(loader)
        for i, data in enumerate(pbar):
            images = data[0].to(args.device, non_blocking=True)
            transcr = data[1]
            s_id = torch.tensor([int(w) for w in data[3]]).to(args.device)

            text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
            text_features = {k: v.to(args.device) for k, v in text_features.items()}

            with autocast(enabled=args.amp):
                if args.latent:
                    with autocast(enabled=False):
                        images_latent = vae.module.encode(images.float()).latent_dist.sample() * 0.18215
                else:
                    images_latent = images

                noise = torch.randn_like(images_latent)
                timesteps = diffusion.sample_timesteps(images_latent.size(0)).to(args.device)
                noisy_images = noise_scheduler.add_noise(images_latent, noise, timesteps)

                drop_labels = np.random.random() < 0.1
                labels = None if drop_labels else s_id
                y = labels if labels is not None else torch.zeros(images_latent.size(0), dtype=torch.long, device=args.device)

                predicted_noise = model(noisy_images, timesteps, text_features, y, style_extractor=None)
                loss = mse_loss(noise, predicted_noise)
                mse_val = loss.item()

                rec_loss_val = None
                if epoch >= args.rec_start_epoch:
                    noise_scheduler.set_timesteps(15)
                    timesteps_list = list(noise_scheduler.timesteps)
                    x_approx = noisy_images.clone()

                    with torch.no_grad():
                        for step in timesteps_list[:-1]:
                            t_approx = (torch.ones(x_approx.size(0), device=args.device) * step).long()
                            pred_noise_approx = model(x_approx, t_approx, text_features, s_id, style_extractor=None)
                            x_approx = noise_scheduler.step(pred_noise_approx.float(), step, x_approx.float()).prev_sample

                    final_step = timesteps_list[-1]
                    t_final = (torch.ones(x_approx.size(0), device=args.device) * final_step).long()
                    pred_noise_final = model(x_approx, t_final, text_features, s_id, style_extractor=None)
                    x_approx = noise_scheduler.step(pred_noise_final.float(), final_step, x_approx.float()).prev_sample

                    # --- THE VARIABLE UNDER TEST ---
                    # control=True reproduces the OLD bug (detach -> no gradient
                    # to UNet) as a same-subset, same-curriculum baseline.
                    # control=False (default) is the fix: no detach.
                    if args.control:
                        x_approx = x_approx.detach()

                    if args.latent:
                        with autocast(enabled=False):
                            image_approx = vae.module.decode((x_approx / 0.18215).float()).sample
                        image_approx = (image_approx / 2 + 0.5).clamp(0, 1)
                    else:
                        image_approx = ((x_approx.clamp(-1, 1) + 1) / 2)

                    pixel_values = preprocess_image_for_recognizer_torch(image_approx, args)
                    gt_labels = pad_sequence(
                        [torch.tensor([_train_mod.recognizer_tokenizer.bos_token_id] +
                                       _train_mod.recognizer_tokenizer(tr).input_ids +
                                       [_train_mod.recognizer_tokenizer.eos_token_id]) for tr in transcr],
                        batch_first=True, padding_value=-100
                    ).to(args.device)

                    with autocast(enabled=False):
                        outputs_emb = _train_mod.recognizer_conv(pixel_values.float())
                        outputs = _train_mod.recognizer_transformer(inputs_embeds=outputs_emb, labels=gt_labels)
                    rec_loss = torch.clamp(outputs.loss, max=5.0)
                    rec_loss_val = rec_loss.item()
                    loss = loss + current_rec_weight * rec_loss

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            ema.step_ema(ema_model, model)

            if i % args.log_every == 0:
                msg = f"  [E{epoch} I{i}] MSE: {mse_val:.6f}"
                if rec_loss_val is not None:
                    msg += f" | Rec (raw): {rec_loss_val:.6f} | Weighted: {current_rec_weight * rec_loss_val:.6f}"
                print(msg)
                results_log['step_log'].append({
                    'epoch': epoch, 'step': i, 'mse': mse_val, 'rec_loss': rec_loss_val,
                    'rec_weight': current_rec_weight
                })

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            tag = f"epoch{epoch}_{'control' if args.control else 'fixed'}"
            result = evaluate_holdout(
                ema_model, vae, diffusion, noise_scheduler, held_out_vocab, num_classes,
                args, tokenizer, text_encoder, os.path.join(args.save_path, 'eval'), tag,
                args.eval_sample_count
            )
            results_log['eval'].append({'epoch': epoch, 'tag': tag, 'result': result})
            with open(os.path.join(args.save_path, 'results_log.json'), 'w', encoding='utf-8') as f:
                json.dump(results_log, f, ensure_ascii=False, indent=2)

    return results_log


def main():
    parser = argparse.ArgumentParser()
    # dataset / paths -- reuse same defaults as main training script
    parser.add_argument('--train_image_folder', type=str, default=r'./Urdu_Word_Dataset/train/images')
    parser.add_argument('--train_gt_folder', type=str, default=r'./Urdu_Word_Dataset/train/gt_txt')
    parser.add_argument('--save_path', type=str, default='./screening_diagnostic_out')
    parser.add_argument('--recognizer_conv_path', type=str, default='/content/DiffusionRec_Research/weights/conv_transformer_weights/icdar/conv.pt')
    parser.add_argument('--recognizer_transformer_path', type=str, default='/content/DiffusionRec_Research/weights/conv_transformer_weights/icdar')
    parser.add_argument('--recognizer_vocab_path', type=str, default='/content/DiffusionRec_Research/data/vocab/ved/')
    parser.add_argument('--stable_dif_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--init_checkpoint', type=str, default=None,
                         help='Path to a previous checkpoint (e.g. ckpt.pt) to fine-tune on top of. '
                              'Leave unset to train from a random init.')
    parser.add_argument('--init_ema_checkpoint', type=str, default=None,
                         help='Optional path to a previous EMA checkpoint (e.g. ema_ckpt.pt) to '
                              'initialize ema_model from directly. If unset, ema_model just starts '
                              'as a copy of the fine-tuned unet -- fine for a short diagnostic, but '
                              'the real EMA weights are a more honest "before" state if you have them.')

    # diagnostic-specific
    parser.add_argument('--subset_size', type=int, default=1000)
    parser.add_argument('--rotate_every', type=int, default=4, help='Resample training subset every N epochs')
    parser.add_argument('--held_out_words', type=int, default=50,
                         help='How many unique words from the (disjoint) val set to use for held-out eval')
    parser.add_argument('--eval_sample_count', type=int, default=50,
                         help='How many held-out words to sample per eval. Defaults to using the full '
                              'held-out set (matches --held_out_words) so accuracy numbers aren\'t noisy '
                              'from too small a sample.')
    parser.add_argument('--qwen_checkpoint', type=str, default=None,
                         help='Path to fine-tuned Qwen3-VL weights (local dir or HF hub id). '
                              'If unset, eval images/manifests are still saved but not scored.')
    parser.add_argument('--qwen_device', type=str, default='auto',
                         help='device_map passed to Qwen3-VL. Loaded fresh right before each eval '
                              'pass and fully deleted afterward (not kept resident between passes, '
                              'GPU or CPU) -- a 4B model sitting idle for epochs at a time was itself '
                              'the cause of an earlier crash (system RAM, not GPU VRAM). "auto" is '
                              'fine here since the model is never moved after loading, only deleted.')
    parser.add_argument('--qwen_prompt_variant', type=str, default='plain',
                         choices=list(qwen_eval.PROMPT_VARIANTS.keys()))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--control', type=str2bool, default=False,
                         help='If True, detach before rec loss (reproduces the OLD bug) as a control run.')
    parser.add_argument('--seed', type=int, default=42)

    # aggressive curriculum defaults -- diagnostic only, not a real recipe
    parser.add_argument('--rec_start_epoch', type=int, default=0)
    parser.add_argument('--rec_weight_start', type=float, default=0.01)
    parser.add_argument('--rec_weight_max', type=float, default=0.15)
    parser.add_argument('--rec_curriculum_epochs', type=int, default=5)

    # standard training args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_size', type=tuple, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--color', type=str2bool, default=True)
    parser.add_argument('--latent', type=str2bool, default=True)
    parser.add_argument('--amp', type=str2bool, default=True)
    parser.add_argument('--sampling_steps', type=int, default=18)

    # ------------------------------------------------------------------
    # The following aren't used directly by this diagnostic script, but
    # UNetModel (and other components reused from the main training
    # script) read them off the `args` namespace internally -- e.g.
    # `self.interpolation = args.interpolation` inside models/unet.py.
    # They need to exist on args even though this script never branches
    # on them itself, or you'll hit AttributeError deep inside a reused
    # class. Defaults match the original training script.
    # ------------------------------------------------------------------
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--dataset', type=str, default='word_generation')
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--img_feat', type=str2bool, default=False)
    parser.add_argument('--interpolation', type=str2bool, default=False)
    parser.add_argument('--dataparallel', type=str2bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--sampling_word', type=str2bool, default=False)
    parser.add_argument('--val_image_folder', type=str, default=r'./Urdu_Word_Dataset/val/images')
    parser.add_argument('--val_gt_folder', type=str, default=r'./Urdu_Word_Dataset/val/gt_txt')
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Seed EVERYTHING before any model is constructed or any randomness is
    # drawn. This is what makes --control True vs --control False (run
    # separately, same --seed) an actually-isolated comparison: without
    # this, the UNet's random weight init, the per-step noise draws
    # (torch.randn_like), timestep sampling, and label-dropout would all
    # differ between the two runs on top of the one line under test.
    # Subset rotation was already seeded via random.Random(args.seed)
    # inside diagnostic_train -- this covers everything else.
    # ---------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    setup_logging(args)
    load_urdu_recognizer(device=args.device, conv_path=args.recognizer_conv_path,
                          transformer_path=args.recognizer_transformer_path,
                          vocab_path=args.recognizer_vocab_path)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = WordGenerationDataset(args.train_image_folder, args.train_gt_folder, 'train',
                                        fixed_size=(64, 256), transforms=transform, args=args)
    print('Full train pool:', len(train_data))

    val_data = WordGenerationDataset(args.val_image_folder, args.val_gt_folder, 'val',
                                      fixed_size=(64, 256), transforms=transform, args=args)
    print('Val pool (genuinely disjoint, never trained on):', len(val_data))

    style_classes = train_data.wclasses
    character_classes = train_data.character_classes
    vocab_size = len(character_classes) + _train_mod.num_tokens

    # Entire train set is fair game for subset rotation -- no carve-out needed,
    # since held-out eval now comes from the disjoint val set instead.
    pool_indices = list(range(len(train_data)))
    held_out_vocab = build_val_held_out_vocab(val_data, args.held_out_words, args.seed)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    device_id = int(''.join(filter(str.isdigit, args.device)) or 0)
    text_encoder = nn.DataParallel(text_encoder, device_ids=[device_id]).to(args.device)

    unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim,
                      out_channels=args.channels, num_res_blocks=args.num_res_blocks,
                      attention_resolutions=(1, 1), channel_mult=(1, 1), num_heads=args.num_heads,
                      num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size,
                      text_encoder=text_encoder, args=args)
    unet = DataParallel(unet, device_ids=[device_id]).to(args.device)

    if args.init_checkpoint:
        if not os.path.exists(args.init_checkpoint):
            raise FileNotFoundError(
                f"--init_checkpoint not found: {args.init_checkpoint}\n"
                f"Double check the path -- e.g. .../word_level_model/models/ckpt.pt"
            )
        print(f"Loading init weights from {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location=args.device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        unet.load_state_dict(state_dict)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    if args.init_ema_checkpoint:
        if not os.path.exists(args.init_ema_checkpoint):
            raise FileNotFoundError(f"--init_ema_checkpoint not found: {args.init_ema_checkpoint}")
        print(f"Loading EMA init weights from {args.init_ema_checkpoint}")
        ema_ckpt = torch.load(args.init_ema_checkpoint, map_location=args.device)
        ema_state_dict = ema_ckpt['ema_model_state_dict'] if 'ema_model_state_dict' in ema_ckpt else ema_ckpt
        ema_model.load_state_dict(ema_state_dict)

    scaler = GradScaler(enabled=args.amp)

    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = DataParallel(vae, device_ids=[device_id]).to(args.device)
    vae.requires_grad_(False)

    ddim = DPMSolverMultistepScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    ddim.config.algorithm_type = "dpmsolver++"

    if not args.qwen_checkpoint:
        print("\n[WARN] --qwen_checkpoint not set -- eval images/manifests will be saved but not "
              "scored. Pass --qwen_checkpoint to get quantitative Rec.Acc./CER numbers.")

    results_log = {
        'args': vars(args),
        'held_out_vocab': held_out_vocab,
        'subset_rotations': [],
        'step_log': [],
        'eval': [],
    }

    # baseline eval BEFORE any diagnostic training, on the same held-out words
    print("\n=== Baseline eval (before diagnostic training) ===")
    baseline_result = evaluate_holdout(
        ema_model, vae, diffusion, ddim, held_out_vocab, style_classes,
        args, tokenizer, text_encoder, os.path.join(args.save_path, 'eval'), 'baseline',
        args.eval_sample_count
    )
    results_log['eval'].append({'epoch': -1, 'tag': 'baseline', 'result': baseline_result})

    print("\n=== Diagnostic training ===")
    results_log = diagnostic_train(
        diffusion, unet, ema, ema_model, vae, optimizer, mse_loss,
        train_data, pool_indices, held_out_vocab, style_classes,
        ddim, args, tokenizer, text_encoder, scaler, results_log
    )

    final_path = os.path.join(args.save_path, 'results_log.json')
    with open(final_path, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Full log saved to {final_path}")
    print("Compare 'baseline' eval to the final 'eval' entry to see whether "
          "recognition accuracy moved on held-out (never-trained-on) words.")


if __name__ == "__main__":
    main()