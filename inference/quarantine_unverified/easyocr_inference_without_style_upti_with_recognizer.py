import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler
from unet import UNetModel
from torchvision import transforms
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
from evaluate import load
import argparse
import random
import copy
import cv2
from scipy import linalg  # For FID
from torchvision.models import inception_v3  # For FID and IS
from torchvision.transforms import Normalize  # For FID
import torch.nn.functional as F  # For interpolation
from nltk.translate.bleu_score import sentence_bleu  # For BLEU
import easyocr  # For EasyOCR

cer_metric = load("cer")
wer_metric = load("wer")
torch.cuda.empty_cache()

OUTPUT_MAX_LEN = 95
IMG_WIDTH = 1024
IMG_HEIGHT = 64

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

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None, reader=None):
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
        recognized_texts = recognize_urdu_batch_easyocr(x, reader)  # Updated to EasyOCR
        print(f"Generated texts recognized: {recognized_texts}")
        return x

# New EasyOCR-based recognizer (replaces Tesseract)
def recognize_urdu_batch_easyocr(image_tensors, reader):
    recognized_texts = []
    for img_tensor in image_tensors:
        # Convert tensor to PIL Image
        if img_tensor.dim() == 3 and img_tensor.shape[0] in [1, 3]:  # Handle grayscale/RGB
            if img_tensor.shape[0] == 3:
                img = torchvision.transforms.ToPILImage()(img_tensor.cpu())
            else:
                img = torchvision.transforms.ToPILImage()(img_tensor.cpu().squeeze(0))
        else:
            raise ValueError("Unexpected image tensor shape")

        # Preprocess: Binarize for better OCR (optional but improves accuracy)
        img_cv = np.array(img)
        # Convert to grayscale if RGB
        if len(img_cv.shape) == 3 and img_cv.shape[2] == 3:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        _, img_bin = cv2.threshold(img_cv, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = Image.fromarray(img_bin)

        # Run EasyOCR: lang_list=['ur'] for Urdu
        result = reader.readtext(np.array(img), detail=0, paragraph=True)
        text = ' '.join(result).strip().replace('\n', ' ')  # Clean up
        recognized_texts.append(text)
    return recognized_texts
    
from utils.upti_dataset_subset import UPTIDataset
from utils.auxilary_functions import *

# Evaluation Helper Functions
def get_inception_model(device):
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    class InceptionFeature(nn.Module):
        def __init__(self, inception):
            super().__init__()
            self.inception = inception
            self.inception.fc = nn.Identity()  # For 2048 features

        def forward(self, x):
            return self.inception(x)

    class InceptionProb(nn.Module):
        def __init__(self, inception):
            super().__init__()
            self.inception = inception

        def forward(self, x):
            return F.softmax(self.inception(x), dim=1)

    return InceptionFeature(inception), InceptionProb(inception)

def preprocess_for_inception(images):
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(images)

def calculate_fid(real_feats, gen_feats):
    mu_r = real_feats.mean(0).cpu().numpy()
    sigma_r = np.cov(real_feats.cpu().numpy(), rowvar=False)
    mu_g = gen_feats.mean(0).cpu().numpy()
    sigma_g = np.cov(gen_feats.cpu().numpy(), rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_g - 2 * covmean)
    return fid

def calculate_is(probs):
    kl_div = probs * (torch.log(probs) - torch.log(probs.mean(0)))
    scores = torch.exp(kl_div.sum(1))
    is_score = scores.mean().item()
    return is_score

def calculate_bleu(rec_texts, gt_texts):
    bleu_scores = []
    for rec, gt in zip(rec_texts, gt_texts):
        ref_tokens = gt.split()  # Split into words
        cand_tokens = rec.split()
        bleu = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu)
    return np.mean(bleu_scores)

def validate(diffusion, model, vae, test_loader, num_classes, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, inception_feat=None, inception_prob=None, reader=None):
    model.eval()
    gt_texts = []
    rec_texts = []
    real_features_list = []
    gen_features_list = []
    gen_probs_list = []
    max_eval_batches = args.max_eval_batches
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= max_eval_batches:
                break
            images = data[0].to(args.device)  # Real images [-1,1]
            transcr = data[1]
            batch_size = len(transcr)
            transcr = transcr[:batch_size]
            gt_texts.extend(transcr)
            labels = torch.arange(batch_size).long().to(args.device) % num_classes
            sampled_images = diffusion.sampling(model, vae, batch_size, transcr, labels, args, None, noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, reader=reader)
            recognized = recognize_urdu_batch_easyocr(sampled_images, reader)  # Updated to EasyOCR
            rec_texts.extend(recognized)

            # Prepare for FID/IS: Convert to [0,1]
            real_imgs = (images + 1) / 2.0  # [-1,1] to [0,1]
            gen_imgs = sampled_images.float() / 255.0 if sampled_images.dtype == torch.uint8 else sampled_images  # Ensure [0,1]

            # Handle channels (repeat if grayscale)
            if real_imgs.shape[1] == 1:
                real_imgs = real_imgs.repeat(1, 3, 1, 1)
            if gen_imgs.shape[1] == 1:
                gen_imgs = gen_imgs.repeat(1, 3, 1, 1)

            # Extract features and probs
            real_feats = inception_feat(preprocess_for_inception(real_imgs))
            gen_feats = inception_feat(preprocess_for_inception(gen_imgs))
            gen_probs = inception_prob(preprocess_for_inception(gen_imgs))
            real_features_list.append(real_feats)
            gen_features_list.append(gen_feats)
            gen_probs_list.append(gen_probs)

            torch.cuda.empty_cache()
    cer_score = cer_metric.compute(predictions=rec_texts, references=gt_texts)
    accuracy_cer = 1 - cer_score
    wer_score = wer_metric.compute(predictions=rec_texts, references=gt_texts)
    bleu_score = calculate_bleu(rec_texts, gt_texts)
    print(f"Validation CER: {cer_score}")
    print(f"Validation Recognition Accuracy: {accuracy_cer}")
    print(f"Validation WER: {wer_score}")
    print(f"Validation BLEU: {bleu_score}")

    # Compute FID
    if real_features_list:
        real_features = torch.cat(real_features_list, dim=0)
        gen_features = torch.cat(gen_features_list, dim=0)
        fid_score = calculate_fid(real_features, gen_features)
        print(f"Validation FID: {fid_score}")

        # Compute IS
        gen_probs = torch.cat(gen_probs_list, dim=0)
        is_score = calculate_is(gen_probs)
        print(f"Validation IS: {is_score}")
    else:
        print("No features collected for FID/IS.")

    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='line')
    parser.add_argument('--img_size', type=tuple, default=(64, 1024))
    parser.add_argument('--dataset', type=str, default='upti2_2')
    parser.add_argument('--image_folder', type=str, default='/storage/1/saima/images_upti2_2/images/test')
    parser.add_argument('--gt_folder', type=str, default='/home/tukl/Documents/Saima/urdu_handwritten_generation/DiffusionPen/groundtruth_upti2/groundtruth/test')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./unhd_diffusion_model/models_with_recognizer_upti')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--stable_dif_path', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--max_eval_batches', type=int, default=125)  # 1000 samples
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    args = parser.parse_args()

    print('torch version', torch.__version__)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_data = UPTIDataset(args.image_folder, args.gt_folder, 'test', 'line', fixed_size=(64, 1024), transforms=transform, args=args)
    print('test data', len(test_data))
    style_classes = test_data.wclasses
    character_classes = test_data.character_classes
    global letter2index
    letter2index = {char: idx for idx, char in enumerate(character_classes)}
    vocab_size = len(character_classes) + num_tokens
    print('num of character classes', vocab_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.dataparallel:
        device_ids = [3, 4]
    else:
        device_ids = [int(''.join(filter(str.isdigit, args.device)))]

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
    text_encoder = text_encoder.to(args.device)

    unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)
    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)

    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    unet.load_state_dict(torch.load(os.path.join(args.save_path, 'ckpt.pt')))
    ema_model.load_state_dict(torch.load(os.path.join(args.save_path, 'ema_ckpt.pt')))
    print('Loaded models')

    if args.latent:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    # Load Inception for FID/IS
    inception_feat, inception_prob = get_inception_model(args.device)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['ur'], gpu=args.device.startswith('cuda'), verbose=False, model_storage_directory='/storage/1/saima/hf-cache/')

    # Inference on specific sentences
    print('Sampling multi sentences')
    sentences = [
        'مخالفت کےباوجود برصغیرکی ملت اسلامیہ دین اسلام ہے اور اسی نظریہ',
        'بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔',
        'لیکن بدقسمتی سےپاکستان بننے کے بعد ہی اس کے اندر ایسے دشمن',
        'میں مصروف ہوگیا بغیراسکا اہتمام کۓ کہ جسکا وہ پھل کھا رہا ہے'
    ]
    s = random.randint(0, style_classes - 1)
    print('Style:', s)
    labels = torch.tensor([s]).long().to(args.device)
    for idx, sentence in enumerate(sentences):
        print('Sentence:', sentence)
        ema_sampled_images = diffusion.sampling(ema_model, vae, n=1, x_text=sentence, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, reader=reader)
        os.makedirs('./inference_samples', exist_ok=True)
        save_images(ema_sampled_images, os.path.join('./inference_samples', f'sentence_{idx+1}_style_{s}.png'), args)

    # Evaluation on test set
    print('Starting evaluation on test set...')
    validate(diffusion, ema_model, vae, test_loader, style_classes, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, inception_feat=inception_feat, inception_prob=inception_prob, reader=reader)

if __name__ == "__main__":
    main()