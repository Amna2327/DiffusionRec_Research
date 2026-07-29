import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
# Assume unet.py defines UNetModel
from unet import UNetModel
import wandb
from torchvision import transforms
# Import word generation dataset
from utils.word_generation_dataset import WordGenerationDataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
from transformers import GenerationConfig
from safetensors.torch import load_file
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch import Tensor
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, GPT2Tokenizer
from transformers import RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import GPT2Config, GPT2LMHeadModel
from evaluate import load

cer_metric = load("cer")
torch.cuda.empty_cache()

OUTPUT_MAX_LEN = 95  # For word-level, this should be sufficient
IMG_WIDTH = 256  # Word-level width
IMG_HEIGHT = 64  # Word-level height

tokens = {"PAD_TOKEN": 52}
num_tokens = len(tokens)

def label_padding(labels, num_tokens, letter2index):
    ll = [letter2index.get(i, 0) for i in labels]
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if num > 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)
    return ll

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, texts=None, **kwargs):
    """
    Save images as grid with optional text labels
    
    Args:
        images: Tensor of images
        path: Save path
        args: Arguments
        texts: Optional list of Urdu text labels for each image
    """
    grid = torchvision.utils.make_grid(images, padding=2, **kwargs)
    if args.latent:
        im = transforms.ToPILImage()(grid)
        im = im.convert('RGB' if args.color else 'L')
    else:
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
    
    # Add text labels if provided
    if texts is not None:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(im)
        
        # Try to load Urdu font, fallback to default
        try:
            # You'll need to provide path to an Urdu font file (e.g., Noto Nastaliq Urdu)
            font = ImageFont.truetype("arial.ttf", 20)  # Replace with Urdu font path
        except:
            font = ImageFont.load_default()
        
        # Calculate positions for text (below each image in grid)
        num_images = len(texts)
        grid_width = im.width
        img_width = grid_width // num_images if num_images > 0 else grid_width
        
        # Add text below each image
        for i, text in enumerate(texts):
            x_pos = i * img_width + 5  # 5px padding from left edge of each cell
            y_pos = im.height - 25     # 25px from bottom
            
            # Draw text with white background for visibility
            text_bbox = draw.textbbox((x_pos, y_pos), text, font=font)
            draw.rectangle(text_bbox, fill='white')
            draw.text((x_pos, y_pos), text, fill='black', font=font)
    
    im.save(path)
    return im

def crop_whitespace_width(img):
    img_gray = np.array(img)
    _, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    return np.array(img.crop((x, y, x + w, y + h)))

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = 0, 0, 0

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        with torch.no_grad():
            if isinstance(x_text, str):
                x_text = [x_text] * n
            if isinstance(x_text, list):
                x_text = x_text[:n]
                n = len(x_text)
            
            text_features = tokenizer(x_text, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
            text_features = {k: v.to(args.device) for k, v in text_features.items()}
            style_features = None
            
            x = torch.randn((n, 4 if args.latent else 3, self.img_size[0] // 8 if args.latent else self.img_size[0], self.img_size[1] // 8 if args.latent else self.img_size[1])).to(args.device)
            noise_scheduler.set_timesteps(50)
            
            for time in noise_scheduler.timesteps:
                t = (torch.ones(n) * time.item()).long().to(args.device)
                noisy_residual = model(x, t, text_features, labels, original_images=None, mix_rate=mix_rate, style_extractor=style_features)
                x = noise_scheduler.step(noisy_residual, time, x).prev_sample
            
            if args.latent:
                latents = x / 0.18215
                image = vae.module.decode(latents).sample
                x = (image / 2 + 0.5).clamp(0, 1)
            else:
                x = ((x.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
        
        model.train()
        recognized_texts = recognize_urdu_batch(x, args)
        print(f"Generated texts recognized: {recognized_texts}")
        return x

# Recognizer Integration
recognizer_conv = None
recognizer_transformer = None
recognizer_tokenizer = None
generation_config = None

def load_urdu_recognizer(conv_path='./conv_transformer_weights/icdar/conv.pt', transformer_path='./conv_transformer_weights/icdar', device='cuda'):
    global recognizer_conv, recognizer_transformer, recognizer_tokenizer, generation_config
    recognizer_tokenizer = GPT2Tokenizer.from_pretrained("./vocab/ved/")
    recognizer_tokenizer.bos_token = '<s>'
    recognizer_tokenizer.eos_token = '</s>'
    recognizer_tokenizer.pad_token = '<pad>'
    recognizer_tokenizer.unk_token = '<unk>'
    vocab_size = recognizer_tokenizer.vocab_size
    recognizer_conv, recognizer_transformer = model_conv_transformer(vocab_size)
    recognizer_conv.load_state_dict(torch.load(conv_path, weights_only=True, map_location=device))
    config_path = os.path.join(transformer_path, 'config_recognizer.json')
    config_dict = json.load(open(config_path))
    config = EncoderDecoderConfig.from_dict(config_dict)
    config.decoder.tie_word_embeddings = True
    recognizer_transformer = EncoderDecoderModel(config=config)
    state_dict = load_file(os.path.join(transformer_path, 'model.safetensors'))
    recognizer_transformer.load_state_dict(state_dict, strict=False)
    recognizer_conv.to(device)
    recognizer_transformer.to(device)
    recognizer_conv.eval()
    recognizer_transformer.eval()
    recognizer_conv.requires_grad_(False)
    recognizer_transformer.requires_grad_(False)
    generation_config = GenerationConfig(
        max_length=256,
        early_stopping=False,
        no_repeat_ngram_size=0,
        length_penalty=1,
        num_beams=4,
        temperature=1,
        bos_token_id=recognizer_tokenizer.bos_token_id,
        eos_token_id=recognizer_tokenizer.eos_token_id,
        pad_token_id=recognizer_tokenizer.pad_token_id
    )
    print("Urdu recognizer loaded.")

def recognize_urdu_batch(image_tensors, args):
    pixel_values = preprocess_image_for_recognizer_torch(image_tensors.to(args.device), args)
    with torch.no_grad():
        outputs_emb = recognizer_conv(pixel_values)
        outputs = recognizer_transformer.generate(inputs_embeds=outputs_emb, generation_config=generation_config)
    recognized_texts = [recognizer_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    return recognized_texts

def preprocess_image_for_recognizer_torch(image_approx, args):
    bs, c, h, w = image_approx.shape
    if c == 3:
        gray = 0.299 * image_approx[:, 0, :, :] + 0.587 * image_approx[:, 1, :, :] + 0.114 * image_approx[:, 2, :, :]
        img = gray.unsqueeze(1)
    else:
        img = image_approx
    img = img.permute(0, 1, 3, 2)
    img = torch.flip(img, dims=[3])
    orig_trans_h, orig_trans_w = img.shape[2], img.shape[3]
    min_xy = min(1600 / orig_trans_h, 64 / orig_trans_w)
    new_trans_h = int(orig_trans_h * min_xy)
    new_trans_w = int(orig_trans_w * min_xy)
    if new_trans_h != orig_trans_h or new_trans_w != orig_trans_w:
        img = F.interpolate(img, size=(new_trans_h, new_trans_w), mode='bilinear', align_corners=False)
    pixel_values = torch.ones((bs, 1, 1600, 64), device=img.device, dtype=img.dtype)
    pixel_values[:, :, :new_trans_h, :new_trans_w] = img
    pixel_values = 1 - pixel_values
    return pixel_values

def model_conv_transformer(vocab_size):
    class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.2),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.LeakyReLU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.2),
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, 4),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(),
            )

        def forward(self, src: Tensor):
            src = self.conv1(src)
            src = self.conv2(src)
            src = self.conv3(src)
            src = self.conv4(src)
            src = self.conv5(src)
            src = src.squeeze(-1)
            src = src.permute((0, 2, 1)).contiguous()
            return src

    model_conv = Conv()
    dec = {'vocab_size': vocab_size, 'n_positions': 512, 'n_embd': 256, 'n_head': 4, 'n_layer': 2}
    enc = {'vocab_size': vocab_size, 'num_hidden_layers': 2, 'hidden_size': 256, 'num_attention_heads': 4, 'intermediate_size': 1024, 'hidden_act': 'gelu'}
    enc_config = RobertaConfig(**enc)
    dec_config = GPT2Config(**dec)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model_transformer = EncoderDecoderModel(config=config)
    return model_conv, model_transformer

def load_checkpoint(checkpoint_path, model, ema_model, optimizer, ema, lr_scheduler=None, device='cuda'):
    """
    Load a complete training checkpoint and restore all states
    
    Returns:
        start_epoch: The epoch to resume training from
        checkpoint: The full checkpoint dict (for any additional info)
    """
    print(f"\n{'='*60}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"{'='*60}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model states
    model.load_state_dict(checkpoint['model_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore EMA step counter
    ema.step = checkpoint['ema_step']
    
    # Restore learning rate scheduler if it exists
    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        print(f"Restored learning rate scheduler state")
    
    # Restore random states for reproducibility (with proper error handling)
    try:
        if 'rng_state' in checkpoint:
            rng_state = checkpoint['rng_state']
            # Handle CPU tensor conversion
            if isinstance(rng_state, torch.Tensor):
                if rng_state.device.type != 'cpu':
                    rng_state = rng_state.cpu()
                torch.set_rng_state(rng_state)
                print(f"Restored PyTorch RNG state")
    except Exception as e:
        print(f"Warning: Could not restore PyTorch RNG state: {e}")
    
    try:
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            cuda_rng_states = checkpoint['cuda_rng_state']
            # Ensure all states are on CPU before setting
            if isinstance(cuda_rng_states, list):
                cuda_rng_states = [s.cpu() if isinstance(s, torch.Tensor) and s.device.type != 'cpu' else s for s in cuda_rng_states]
            torch.cuda.set_rng_state_all(cuda_rng_states)
            print(f"Restored CUDA RNG state")
    except Exception as e:
        print(f"Warning: Could not restore CUDA RNG state: {e}")
    
    try:
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
            print(f"Restored NumPy RNG state")
    except Exception as e:
        print(f"Warning: Could not restore NumPy RNG state: {e}")
    
    try:
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
            print(f"Restored Python RNG state")
    except Exception as e:
        print(f"Warning: Could not restore Python RNG state: {e}")
    
    # Get the epoch to resume from
    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    
    print(f"\nCheckpoint loaded successfully!")
    print(f"Resuming from epoch: {start_epoch}")
    print(f"Previous loss average: {checkpoint.get('loss_meter_avg', 'N/A')}")
    print(f"EMA step: {ema.step}")
    print(f"{'='*60}\n")
    
    return start_epoch, checkpoint
def gpu_mem(prefix="", device=None):
    if not torch.cuda.is_available():
        return ""

    device = device or torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    max_alloc = torch.cuda.max_memory_allocated(device) / 1024**2

    return (f"{prefix} GPU | "
            f"Alloc: {allocated:.1f} MB | "
            f"Res: {reserved:.1f} MB | "
            f"Max: {max_alloc:.1f} MB")

def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, val_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None, letter2index=None, start_epoch=0):
    model.train()
    loss_meter = AvgMeter()
    print(f'Training started from epoch {start_epoch}....')
    for epoch in range(start_epoch, args.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        # Curriculum for rec_weight
        current_rec_weight = min(args.rec_weight_start + (args.rec_weight_max - args.rec_weight_start) * (epoch / args.rec_curriculum_epochs), args.rec_weight_max)
        print(f'Epoch: {epoch}, Current Rec Weight: {current_rec_weight}')
        pbar = tqdm(loader)
        
        try:
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                s_id = torch.tensor([int(w) for w in data[3]]).to(args.device)
                style_images = data[7].to(args.device)
                
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
                text_features = {k: v.to(args.device) for k, v in text_features.items()}
                
                style_features = None
                
                if args.latent:
                    images = vae.module.encode(images.float()).latent_dist.sample() * 0.18215
                
                noise = torch.randn_like(images)
                timesteps = diffusion.sample_timesteps(images.size(0)).to(args.device)
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                
                drop_labels = np.random.random() < 0.1
                labels = None if drop_labels else s_id
                y = labels if labels is not None else torch.zeros(images.size(0), dtype=torch.long, device=args.device)
                
                predicted_noise = model(noisy_images, timesteps, text_features, y, style_extractor=style_features)
                loss = mse_loss(noise, predicted_noise)
                
                # Rec loss (with fixes)
                if i % 50 == 0 and epoch >= args.rec_start_epoch:
                    print(gpu_mem(prefix=f"[E{epoch} I{i}] Before Rec"))
                    noise_scheduler.set_timesteps(15)
                    x_approx = noisy_images.clone()
                    for step in noise_scheduler.timesteps:
                        t_approx = (torch.ones(x_approx.size(0), device=args.device) * step).long()
                        y_rec = s_id
                        pred_noise_approx = model(x_approx, t_approx, text_features, y_rec, style_extractor=style_features)
                        x_approx = noise_scheduler.step(pred_noise_approx, step, x_approx).prev_sample
                    
                    x_approx = x_approx.detach()
                    
                    if args.latent:
                        image_approx = vae.module.decode(x_approx / 0.18215).sample
                        image_approx = (image_approx / 2 + 0.5).clamp(0, 1)
                    else:
                        image_approx = ((x_approx.clamp(-1, 1) + 1) / 2)
                    
                    pixel_values = preprocess_image_for_recognizer_torch(image_approx, args)
                    gt_labels = pad_sequence([torch.tensor([recognizer_tokenizer.bos_token_id] + recognizer_tokenizer(tr).input_ids + [recognizer_tokenizer.eos_token_id]) for tr in transcr], batch_first=True, padding_value=-100).to(args.device)
                    outputs_emb = recognizer_conv(pixel_values)
                    outputs = recognizer_transformer(inputs_embeds=outputs_emb, labels=gt_labels)
                    rec_loss = outputs.loss
                    rec_loss = torch.clamp(rec_loss, max=5.0)
                    print(gpu_mem(prefix=f"[E{epoch} I{i}] After Rec"))
                    
                    if i % 50 == 0:
                        mse_val = mse_loss(noise, predicted_noise).item()
                        print(f"  [Step {i}] MSE: {mse_val:.6f}, Rec: {rec_loss.item():.6f}, "
                            f"Weighted Rec: {(current_rec_weight * rec_loss).item():.6f}")
                    
                    loss += current_rec_weight * rec_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.step_ema(ema_model, model)
                loss_meter.update(loss.item(), images.size(0))
                pbar.set_postfix(MSE=loss_meter.avg)
                if lr_scheduler:
                    lr_scheduler.step()
        
        except RuntimeError as e:
            if "DataLoader worker" in str(e):
                print(f"\n⚠️  DataLoader worker crashed at epoch {epoch}")
                print(f"Error: {e}")
                print(f"This is a known Windows multiprocessing issue.")
                print(f"Saving emergency checkpoint and continuing...")
                
                # Save emergency checkpoint
                emergency_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_model_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ema_step': ema.step,
                    'loss_meter_avg': loss_meter.avg,
                    'loss_meter_sum': loss_meter.sum,
                    'loss_meter_count': loss_meter.count,
                    'args': vars(args),
                }
                emergency_path = os.path.join(args.save_path, "models", f"emergency_checkpoint_epoch_{epoch}.pt")
                torch.save(emergency_checkpoint, emergency_path)
                print(f"✓ Emergency checkpoint saved to {emergency_path}")
                print(f"Recommendation: Restart training with --num_workers 0")
                print(f"Skipping to next epoch...\n")
                continue  # Skip to next epoch
            else:
                raise  # Re-raise if it's a different error
        
        # Sampling
        print(gpu_mem(prefix=f"[Epoch {epoch} PEAK]"))
        if epoch % 1 == 0:
            val_batch = next(iter(val_loader))
            val_transcr = val_batch[1]
            n = min(4, len(val_transcr))
            labels = torch.arange(n).long().to(args.device) % num_classes
            
            # Sample some words
            ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=val_transcr[:n], labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch}_ema.jpg"), args,texts=val_transcr[:n])
            torch.cuda.empty_cache()
            
            if args.wandb_log:
                words_caption = " | ".join(val_transcr[:n])
                caption = f"Epoch {epoch} - With words: {words_caption}"
                wandb.log({"Sampled images": wandb.Image(ema_sampled_images[0], caption=caption)})
        
        if epoch % 1==0:  # Save every epoch
            # Save comprehensive checkpoint with all training state
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_step': ema.step,
                'loss_meter_avg': loss_meter.avg,
                'loss_meter_sum': loss_meter.sum,
                'loss_meter_count': loss_meter.count,
                'args': vars(args),  # Save all arguments for reference
                # Random states for reproducibility (ensure CPU tensors)
                'rng_state': torch.get_rng_state().cpu(),
                'cuda_rng_state': [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
                'numpy_rng_state': np.random.get_state(),
                'python_rng_state': random.getstate(),
            }
            
            # Add lr_scheduler state if it exists
            if lr_scheduler is not None:
                checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
            
            # Save checkpoint
            # checkpoint_path = os.path.join(args.save_path, "models", f"checkpoint_epoch_{epoch}.pt")
            # torch.save(checkpoint, checkpoint_path)
            # print(f"Saved checkpoint at epoch {epoch}")
            
            # Also save as latest checkpoint (for easy resuming)
            latest_checkpoint_path = os.path.join(args.save_path, "models", "latest_checkpoint.pt")
            torch.save(checkpoint, latest_checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch}")
            
            # Keep backward compatibility - save individual files too
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))

        if epoch % 2==0:
            print(f"\nRunning CER validation at epoch {epoch}...")
            validate(
            diffusion=diffusion,
            model=ema_model,            # <-- IMPORTANT: use EMA model
            vae=vae,
            data_loader=val_loader,
            num_classes=num_classes,
            noise_scheduler=noise_scheduler,
            transforms=transforms,
            args=args,
            tokenizer=tokenizer,
            text_encoder=text_encoder)
            torch.cuda.empty_cache()

def validate(diffusion, model, vae, data_loader, num_classes, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None):
    model.eval()
    gt_texts = []
    rec_texts = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= 10:
                break
            images = data[0].to(args.device)
            transcr = data[1]
            batch_size = min(4, len(transcr))
            transcr = transcr[:batch_size]
            gt_texts.extend(transcr)
            labels = torch.arange(batch_size).long().to(args.device) % num_classes
            sampled_images = diffusion.sampling(model, vae, batch_size, transcr, labels, args, None, noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            recognized = recognize_urdu_batch(sampled_images, args)
            rec_texts.extend(recognized)
            torch.cuda.empty_cache()
    
    cer_score = cer_metric.compute(predictions=rec_texts, references=gt_texts)
    print(f"Validation CER: {cer_score}")
    if args.wandb_log:
        wandb.log({"val_cer": cer_score})
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)  # Increased for word-level
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (use 0 for Windows to avoid crashes)')
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')  # Changed to word
    parser.add_argument('--img_size', type=tuple, default=(64, 256))  # Word-level size
    parser.add_argument('--dataset', type=str, default='word_generation')
    # Separate folders for train/val/test
    parser.add_argument('--train_image_folder', type=str, default=r'.\Urdu_Word_Dataset\train\processed_images')
    parser.add_argument('--train_gt_folder', type=str, default=r'.\Urdu_Word_Dataset\train\gt_txt')
    parser.add_argument('--val_image_folder', type=str, default=r'.\Urdu_Word_Dataset\val\processed_images')
    parser.add_argument('--val_gt_folder', type=str, default=r'.\Urdu_Word_Dataset\val\gt_txt')
    parser.add_argument('--test_image_folder', type=str, default=r'.\Urdu_Word_Dataset\test\processed_images')
    parser.add_argument('--test_gt_folder', type=str, default=r'.\Urdu_Word_Dataset\test\gt_txt')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default=r'.\word_level_model')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--resume_training', type=bool, default=True, help='Automatically resume from latest checkpoint if available')
    # parser.add_argument('--checkpoint_freq', type=int, default=20, help='Save checkpoint every N epochs')
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--stable_dif_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--rec_start_epoch', type=int, default=50)
    parser.add_argument('--rec_weight_start', type=float, default=0.001)
    parser.add_argument('--rec_weight_max', type=float, default=0.05)
    parser.add_argument('--rec_curriculum_epochs', type=int, default=150)
    args = parser.parse_args()
    
    print('torch version', torch.__version__)
    if args.wandb_log:
        wandb.init(project='WordGeneration', name=args.dataset, config=args)
    
    setup_logging(args)
    load_urdu_recognizer(device=args.device)
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load word-level dataset
    train_data = WordGenerationDataset(args.train_image_folder, args.train_gt_folder, 'train', fixed_size=(64, 256), transforms=transform, args=args)
    val_data = WordGenerationDataset(args.val_image_folder, args.val_gt_folder, 'val', fixed_size=(64, 256), transforms=transform, args=args)
    test_data = WordGenerationDataset(args.test_image_folder, args.test_gt_folder, 'test', fixed_size=(64, 256), transforms=transform, args=args)
    
    print('train data', len(train_data), 'val data', len(val_data), 'test data', len(test_data))
    
    style_classes = train_data.wclasses
    character_classes = train_data.character_classes
    global letter2index
    letter2index = {char: idx for idx, char in enumerate(character_classes)}
    vocab_size = len(character_classes) + num_tokens
    print('num of character classes', vocab_size)
    
    # DataLoader configuration - Windows-safe settings
    # Use persistent_workers to avoid worker respawning issues
    num_workers = args.num_workers if args.num_workers > 0 else 0
    use_persistent = num_workers > 0  # Only use persistent workers if workers > 0
    
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=use_persistent,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    if args.dataparallel:
        device_ids = [3, 4]
    else:
        device_ids = [int(''.join(filter(str.isdigit, args.device)))]
    
    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = None
        text_encoder = None
    
    if args.unet == 'unet_latent':
        unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)
        unet = DataParallel(unet, device_ids=device_ids)
        unet = unet.to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    # Initialize start_epoch for training
    start_epoch = 0
    
    # Check for resume training (automatic if resume_training=True)
    latest_checkpoint_path = os.path.join(args.save_path, "models", "latest_checkpoint.pt")
    
    if args.resume_training and os.path.exists(latest_checkpoint_path):
        # Automatic resume from latest checkpoint
        try:
            start_epoch, checkpoint = load_checkpoint(
                latest_checkpoint_path, 
                unet, 
                ema_model, 
                optimizer, 
                ema, 
                lr_scheduler, 
                args.device
            )
            print(f"✓ Successfully resumed training from epoch {start_epoch}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
    
    elif args.load_check:
        # Legacy loading (backward compatibility)
        try:
            unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
            optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
            ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
            print('Loaded models and optimizer (legacy mode - epoch info not available)')
            start_epoch = 0  # Can't determine epoch from legacy checkpoints
        except Exception as e:
            print(f"Warning: Failed to load legacy checkpoint: {e}")
            print("Starting training from scratch...")
            start_epoch = 0
    else:
        print("Starting fresh training (no checkpoint found or resume disabled)")
        start_epoch = 0
    
    if args.latent:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None
    
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    feature_extractor = None
    
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, val_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler, letter2index=letter2index, start_epoch=start_epoch)
    elif args.train_mode == 'sampling':
        print('Sampling started....')
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.eval()
        
        # Sample some Urdu words
        words = ['کون', 'سوچ', 'ملک', 'دین']
        s = 0
        labels = torch.tensor([s]).long().to(args.device)
        
        for word in words:
            print('Word:', word)
            sample_image = diffusion.sampling(ema_model, vae, n=1, x_text=word, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            save_images(sample_image, os.path.join('./word_samples/', f'{word}_style_{s}.png'), args)

if __name__ == "__main__":
    main()