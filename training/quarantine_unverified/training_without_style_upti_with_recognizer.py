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
# Assume feature_extractor.py defines ImageEncoder
# from feature_extractor import ImageEncoder
from utils.upti_dataset_subset import UPTIDataset  # Assume this defines dataset with character_classes
from utils.auxilary_functions import *  # Assume auxiliary functions
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
# Assume tacobox installed
# from tacobox import Taco
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

OUTPUT_MAX_LEN = 95  # +2 for <GO> + groundtruth + <END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens, letter2index):  # Added letter2index param
    ll = [letter2index.get(i, 0) for i in labels]  # Use .get to handle missing
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if num > 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)
    return ll

tokens = {"PAD_TOKEN": 52}  # Simplified; adjust if needed
num_tokens = len(tokens)

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent:
        im = transforms.ToPILImage()(grid)
        im = im.convert('RGB' if args.color else 'L')
    else:
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
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
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 1024), args=None):
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

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        all_generated = []
        with torch.no_grad():
            pbar = tqdm(test_loader)
            for data in pbar:  # Fixed: Loop over all batches
                images = data[0].to(args.device)
                transcr = data[1]
                labels = torch.tensor([int(w) for w in data[3]]).to(args.device)  # Use data[3] for labels
                style_images = data[7].to(args.device)
                if args.model_name == 'wordstylist':
                    batch_word_embeddings = [torch.from_numpy(np.array(label_padding(trans, num_tokens, letter2index), dtype="int64")).long().to(args.device) for trans in transcr]
                    text_features = torch.stack(batch_word_embeddings)
                else:
                    text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
                    text_features = {k: v.to(args.device) for k, v in text_features.items()}
                style_features = None
                #x = torch.randn((images.size(0), 4 if args.latent else 3, *(self.img_size[::-1] if args.latent else self.img_size))).to(args.device)  # Simplified
                x = torch.randn((images.size(0), 4 if args.latent else 3, self.img_size[0] // 8 if args.latent else self.img_size[0], self.img_size[1] // 8 if args.latent else self.img_size[1])).to(args.device)
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:
                    t = (torch.ones(images.size(0)) * time.item()).long().to(args.device)
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    x = noise_scheduler.step(noisy_residual, time, x).prev_sample
                if args.latent:
                    latents = x / 0.18215  # Fixed scaling
                    image = vae.module.decode(latents).sample
                    image = (image / 2 + 0.5).clamp(0, 1)
                    x = image
                else:
                    x = ((x.clamp(-1, 1) + 1) / 2 * 255).type(torch.uint8)
                all_generated.append(x)
        model.train()
        all_generated = torch.cat(all_generated, dim=0)  # Concat all
        recognized_texts = recognize_urdu_batch(all_generated, args)
        print(f"Generated texts recognized: {recognized_texts}")
        return all_generated

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        with torch.no_grad():
            if isinstance(x_text, str):
                x_text = [x_text] * n
            # Ensure x_text is a list and limit batch size to prevent OOM
            if isinstance(x_text, list):
                x_text = x_text[:n]
                n = len(x_text)  # Update n to actual list length
            # Properly tokenize and move to device
            text_features = tokenizer(x_text, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
            text_features = {k: v.to(args.device) for k, v in text_features.items()}
            style_features = None
            #x = torch.randn((n, 4 if args.latent else 3, *(self.img_size[::-1] if args.latent else self.img_size))).to(args.device)
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
class HWRDataset(Dataset):
    def __init__(self, df, tokenizer, input_width=1600, input_height=64, aug=False, taco_aug_frac=0.9):
        self.df = df
        self.input_width = input_width
        self.input_height = input_height
        self.tokenizer = tokenizer
        self.mytaco = Taco(cp_vertical=0.2, cp_horizontal=0.25, max_tw_vertical=100, min_tw_vertical=10, max_tw_horizontal=50, min_tw_horizontal=10)
        self.aug = aug
        self.taco_aug_frac = taco_aug_frac

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        pixel_values = self.preprocess(image, self.aug)
        labels = self.tokenizer(text).input_ids
        labels = [self.tokenizer.bos_token_id] + [label if label != self.tokenizer.pad_token_id else -100 for label in labels] + [self.tokenizer.eos_token_id]
        return torch.tensor(pixel_values[None, :, :]).float(), torch.tensor(labels)

    def preprocess(self, img, augment=True):
        if augment:
            img = self.apply_taco_augmentations(img)
        img = img / 255
        img = img.swapaxes(-2, -1)[..., ::-1]
        target = np.ones((self.input_width, self.input_height))
        new_x = self.input_width / img.shape[0]
        new_y = self.input_height / img.shape[1]
        min_xy = min(new_x, new_y)
        new_x = int(img.shape[0] * min_xy)
        new_y = int(img.shape[1] * min_xy)
        img2 = cv2.resize(img, (new_y, new_x))
        target[:new_x, :new_y] = img2
        return 1 - target

    def apply_taco_augmentations(self, input_img):
        random_value = random.random()
        if random_value <= self.taco_aug_frac:
            augmented_img = self.mytaco.apply_vertical_taco(input_img, corruption_type='random')
        else:
            augmented_img = input_img
        return augmented_img

def collate_fn(batch):
    src_batch, tgt_batch = list(zip(*batch))
    src_batch = torch.stack(src_batch)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=-100)
    return {'pixel_values': src_batch, 'labels': tgt_batch}

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

def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None, letter2index=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    for epoch in range(args.epochs):
        # Curriculum for rec_weight
        current_rec_weight = min(args.rec_weight_start + (args.rec_weight_max - args.rec_weight_start) * (epoch / args.rec_curriculum_epochs), args.rec_weight_max)
        print(f'Epoch: {epoch}, Current Rec Weight: {current_rec_weight}')
        pbar = tqdm(loader)
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = torch.tensor([int(w) for w in data[3]]).to(args.device)
            style_images = data[7].to(args.device)
            if args.model_name == 'wordstylist':
                batch_word_embeddings = [torch.from_numpy(np.array(label_padding(trans, num_tokens, letter2index), dtype="int64")).long().to(args.device) for trans in transcr]
                text_features = torch.stack(batch_word_embeddings).to(args.device)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200)
                text_features = {k: v.to(args.device) for k, v in text_features.items()}
            style_features = None
            if args.latent:
                images = vae.module.encode(images.float()).latent_dist.sample() * 0.18215
            noise = torch.randn_like(images)
            timesteps = diffusion.sample_timesteps(images.size(0)).to(args.device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            drop_labels = np.random.random() < 0.1  # Flag for dropping
            labels = None if drop_labels else s_id
            y = labels if labels is not None else torch.zeros(images.size(0), dtype=torch.long, device=args.device)
            predicted_noise = model(noisy_images, timesteps, text_features, y, style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise)
            # Rec loss (with fixes)
            if i % 50 == 0 and epoch >= args.rec_start_epoch:
                noise_scheduler.set_timesteps(15)  # Increased from 2 to 15 steps
                x_approx = noisy_images.clone()
                for step in noise_scheduler.timesteps:
                    t_approx = (torch.ones(x_approx.size(0), device=args.device) * step).long()
                    # Force no label drop for rec loss
                    y_rec = s_id  
                    pred_noise_approx = model(x_approx, t_approx, text_features, y_rec, style_extractor=style_features)
                    x_approx = noise_scheduler.step(pred_noise_approx, step, x_approx).prev_sample
                x_approx = x_approx.detach()  # Detach to stop backprop through loop
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
                rec_loss = torch.clamp(rec_loss, max=5.0)  # Prevent explosion
                # Log losses for debugging
                if i % 50 == 0:
                    mse_val = mse_loss(noise, predicted_noise).item()
                    print(f"  [Step {i}] MSE: {mse_val:.6f}, Rec: {rec_loss.item():.6f}, "
                          f"Weighted Rec: {(current_rec_weight * rec_loss).item():.6f}")
                loss += current_rec_weight * rec_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Added clipping
            optimizer.step()
            ema.step_ema(ema_model, model)
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(MSE=loss_meter.avg)
            if lr_scheduler:
                lr_scheduler.step()
        # Sampling and validation
        if epoch % 1 == 0:
            # Get first batch from test_loader for sampling (limit to 4 to prevent OOM)
            test_batch = next(iter(test_loader))
            test_transcr = test_batch[1]
            n = min(4, len(test_transcr))  # Limit to 4 samples to prevent OOM
            labels = torch.arange(n).long().to(args.device) % num_classes
            if args.sampling_word:
                words = ['text']
                for x_text in words:
                    ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                    save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch}_ema.jpg"), args)
            else:
                # Sample using first batch transcriptions instead of entire test set
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=test_transcr[:n], labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch}_ema.jpg"), args)
            torch.cuda.empty_cache()  # Clear cache after sampling
            if args.wandb_log:
                wandb.log({"Sampled images": wandb.Image(ema_sampled_images[0], caption=f"Epoch {epoch}")})  # Simplified
            # Urdu samples (unchanged)
            urdu_text = 'کون سوچ سکتا تھا کہ ہندوستان اکثریت اورانگریزحکمرانوں کی مشترکہ'
            s = 0
            labels = torch.tensor([s]).long().to(args.device)
            sample_image = diffusion.sampling(ema_model, vae, n=1, x_text=urdu_text, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            save_images(sample_image, os.path.join(args.save_path, 'images', f'urdu_sample_epoch_{epoch}_style_{s}.jpg'), args)
            urdu_text2 = 'لیکن بدقسمتی سےپاکستان بننے کے بعد ہی اس کے اندر ایسے دشمن'
            s = 0 # fixed style
            labels = torch.tensor([s]).long().to(args.device)
            sample_image2 = diffusion.sampling(ema_model, vae, 1, urdu_text2, labels, args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            save_images(sample_image2, os.path.join(args.save_path, 'images', f'urdu_sample2_epoch_{epoch}_style_{s}.jpg'), args)
        # TEMPORARILY DISABLED TO TEST OOM
        if epoch % 1 == 0:
            pass  # validate(diffusion, ema_model, vae, test_loader, num_classes, noise_scheduler, transforms, args, tokenizer, text_encoder)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))

def validate(diffusion, model, vae, test_loader, num_classes, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None):
    model.eval()
    gt_texts = []
    rec_texts = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # Only validate on first 10 batches to prevent OOM
            if i >= 10:
                break
            images = data[0].to(args.device)
            transcr = data[1]
            # Limit to 4 samples per batch to prevent OOM
            batch_size = min(4, len(transcr))
            transcr = transcr[:batch_size]
            gt_texts.extend(transcr)
            labels = torch.arange(batch_size).long().to(args.device) % num_classes
            sampled_images = diffusion.sampling(model, vae, batch_size, transcr, labels, args, None, noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            recognized = recognize_urdu_batch(sampled_images, args)
            rec_texts.extend(recognized)
            # Clear cache after each batch to prevent OOM
            torch.cuda.empty_cache()
    cer_score = cer_metric.compute(predictions=rec_texts, references=gt_texts)
    print(f"Validation CER: {cer_score}")
    if args.wandb_log:
        wandb.log({"val_cer": cer_score})
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='line')
    parser.add_argument('--img_size', type=tuple, default=(64, 256))
    parser.add_argument('--dataset', type=str, default='upti2_2')
    parser.add_argument('--image_folder', type=str, default='./images_upti2_2/images/train/')
    parser.add_argument('--gt_folder', type=str, default='./groundtruth_upti2/groundtruth/train/')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_upti_model_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/upti_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--rec_start_epoch', type=int, default=50, help='Epoch to start adding recognition loss')
    parser.add_argument('--rec_weight_start', type=float, default=0.001, help='Starting weight for recognition loss (reduced from 0.01)')
    parser.add_argument('--rec_weight_max', type=float, default=0.05, help='Max weight for recognition loss (reduced from 0.1)')
    parser.add_argument('--rec_curriculum_epochs', type=int, default=150, help='Epochs for rec weight curriculum (increased from 100)')
    args = parser.parse_args()
    print('torch version', torch.__version__)
    if args.wandb_log:
        wandb.init(project='DiffScribe', name=args.dataset, config=args)
    setup_logging(args)
    load_urdu_recognizer(device=args.device)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = UPTIDataset(args.image_folder, args.gt_folder, 'train', 'line', fixed_size=(64, 1024), transforms=transform, args=args)
    test_data = UPTIDataset(args.image_folder.replace('train', 'test'), args.gt_folder.replace('train', 'test'), 'test', 'line', fixed_size=(64, 1024), transforms=transform, args=args)  # Fixed: Proper test set
    print('train data', len(train_data), 'test data', len(test_data))
    style_classes = train_data.wclasses
    character_classes = train_data.character_classes  # Assume this is list of chars
    global letter2index  # Mock fix; adapt to your dataset
    letter2index = {char: idx for idx, char in enumerate(character_classes)}  # Example dict
    vocab_size = len(character_classes) + num_tokens
    print('num of character classes', vocab_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)  # Full test
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
    if args.load_check:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
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
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler, letter2index=letter2index)
    elif args.train_mode == 'sampling':
        print('Sampling started....')
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.eval()
        if args.sampling_mode == 'single_sampling':
            x_text = ['کون', 'سوچ'] # Urdu text examples
            for x_text in x_text:
                print('Word:', x_text)
                s = random.randint(0, style_classes - 1) # index for style class
                print('style', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
                save_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)
        elif args.sampling_mode == 'paragraph':
            print('Sampling paragraph')
            #make the code to generate lines
            lines = 'کون سوچ سکتا تھا کہ ہندوستان اکثریت اورانگریزحکمرانوں کی مشترکہ مخالفت کےباوجود برصغیرکی ملت اسلامیہ دین اسلام ہے اور اسی نظریہ پر اس ملک میں بسنے والے مختلف عناصر کا اتحاد ہےاورپاکستان کی بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔' # Example Urdu text
            fakes= []
            gap = np.ones((64, 16)) * 255
            max_line_width = 900
            total_char_count = 0
            avg_char_width = 0
            current_line_width = 0
            words = lines.strip().split(' ')
            longest_word_length = max(len(word) for word in words)
            #print('longest_word_length', longest_word_length)
            s = random.randint(0, style_classes - 1)#.long().to(args.device)
            print('Style:', s)
            words = words[::-1] # Reverse for RTL
            for word in words:
                print('Word:', word)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
                image = ema_sampled_images.squeeze(0)
                im = torchvision.transforms.ToPILImage()(image)
                im = im.convert("L")
                im = crop_whitespace_width(im)
                im = Image.fromarray(im)
                if len(word) == longest_word_length:
                    max_word_length_width = im.width
                # Calculate aspect ratio
                aspect_ratio = im.width / im.height
                im = np.array(im)
                fakes.append(im)
            # Calculate the scaling factor based on the longest word
            #find the average character width of the max length word
            avg_char_width = max_word_length_width / longest_word_length
            print('avg_char_width', avg_char_width)
            #scaling_factor = avg_char_width / (32 * aspect_ratio) # Aspect ratio of an average character
            # Scale and pad each word
            scaled_padded_words = []
            max_height = 64 # Defined max height for all images
            punctuation = '.,?!:;،۔؟' # Added Urdu punctuation
            for word, img in zip(words, fakes):
                img_pil = Image.fromarray(img)
                as_ratio = img_pil.width / img_pil.height
                #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
                scaled_width = int(avg_char_width * len(word))
                scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
                print(f'Word {word} - scaled_img {scaled_img.size}')
                # Padding
                #if word is in punctuation:
                if any(char in punctuation for char in word):
                    #rescale to height 10
                    w_punc = scaled_img.width
                    h_punc = scaled_img.height
                    as_ratio_punct = w_punc / h_punc
                    if '.' in word or '۔' in word:
                        scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
                    else:
                        scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
                    #pad on top and leave the image in the bottom
                    padding_bottom = 10
                    padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
                    # No padding at the bottom
                    # Apply padding
                    padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
                else:
                    if scaled_img.height < max_height:
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    else:
                        #resize to max height while maintaining aspect ratio
                        #ar = scaled_img.width / scaled_img.height
                        scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    #padded_img = np.array(scaled_img)
                    #print('padded_img', padded_img.shape)
                scaled_padded_words.append(padded_img)
            # Create a gap array (white space)
            height = 64 # Fixed height for all images
            gap = np.ones((height, 16), dtype=np.uint8) * 255 # White gap
            # Concatenate images with gaps
            sentence_img = gap # Start with a gap
            lines = []
            line_img = gap
            ''' sentence_img = gap # Start with a gap
            for img in scaled_padded_words:
                #print('img', img.shape)
                sentence_img = np.concatenate((sentence_img, img, gap), axis=1) '''
            for img in scaled_padded_words:
                img_width = img.shape[1] + gap.shape[1]
                if current_line_width + img_width < max_line_width:
                    # Add the image to the current line
                    if line_img.shape[1] == 0:  # Fixed: Check width
                        line_img = np.ones((height, 0), dtype=np.uint8) * 255 # Start a new line
                    line_img = np.concatenate((line_img, img, gap), axis=1)
                    current_line_width += img_width #+ gap.shape[1]
                    #print('current_line_width if', current_line_width)
                    # Check if adding this image exceeds the max line width
                else:
                    # Pad the current line with white space to max_line_width
                    remaining_width = max_line_width - current_line_width
                    line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                    lines.append(line_img)
                    # Start a new line with the current word
                    line_img = np.concatenate((gap, img, gap), axis=1)
                    current_line_width = img_width #+ 2 * gap.shape[1]
                    #print('current_line_width else', current_line_width)
            # Add the last line to the lines list if current_line_width > 0
            if current_line_width > 0:
                # Pad the last line to max_line_width
                remaining_width = max_line_width - current_line_width
                line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                lines.append(line_img)
            # # Concatenate all lines to form a paragraph, pad them if necessary
            # max_height = max([line.shape[0] for line in lines])
            # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
            # for line in lines:
            # if line.shape[0] < max_height:
            # padding = (max_height - line.shape[0]) // 2
            # line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)
            # #print the shapes
            # print('line shape', line.shape)
            #print('paragraph shape', paragraph_img.shape)
            paragraph_img = np.concatenate(lines, axis=0)
            paragraph_image = Image.fromarray(paragraph_img)
            paragraph_image = paragraph_image.convert("L")
            paragraph_image.save(f'paragraph_style_{s}.png')
        elif args.sampling_mode == 'multi_sentence':
            print('Sampling multi sentences')
            sentences = [
                'مخالفت کےباوجود برصغیرکی ملت اسلامیہ دین اسلام ہے اور اسی نظریہ',
                'بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔',
                'لیکن بدقسمتی سےپاکستان بننے کے بعد ہی اس کے اندر ایسے دشمن',
                'میں مصروف ہوگیا بغیراسکا اہتمام کۓ کہ جسکا وہ پھل کھا رہا ہے'
            ]
            s = random.randint(0, style_classes - 1) # One random style for all
            print('Style:', s)
            labels = torch.tensor([s]).long().to(args.device)
            for idx, sentence in enumerate(sentences):
                print('Sentence:', sentence)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=1, x_text=sentence, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
                save_images(ema_sampled_images, os.path.join(f'./image_samples/', f'sentence_{idx+1}_style_{s}.png'), args)

if __name__ == "__main__":
    main()
