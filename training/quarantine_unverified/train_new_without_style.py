# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# from torch.utils.data import DataLoader, random_split
# import torchvision
# from tqdm import tqdm
# from torch import optim
# import copy
# import argparse
# import uuid
# import json
# from diffusers import AutoencoderKL, DDIMScheduler
# import random
# from unet import UNetModel
# import wandb
# from torchvision import transforms
# from feature_extractor import ImageEncoder
# from utils.unhd_dataset import UNHDDataset # Import UNHD dataset
# from utils.auxilary_functions import *
# from torchvision.utils import save_image
# from torch.nn import DataParallel
# from transformers import CanineModel, CanineTokenizer
# torch.cuda.empty_cache()
# OUTPUT_MAX_LEN = 95 # +2 # <GO>+groundtruth+<END>
# IMG_WIDTH = 1024
# IMG_HEIGHT = 64
# # For Urdu, character classes are auto-computed in dataset
# # No fixed c_classes, use dataset.character_classes
# ### Borrowed from GANwriting ###
# def label_padding(labels, num_tokens):
#     new_label_len = []
#     ll = [letter2index[i] for i in labels]
#     new_label_len.append(len(ll) + 2)
#     ll = np.array(ll) + num_tokens
#     ll = list(ll)
#     # ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
#     num = OUTPUT_MAX_LEN - len(ll)
#     if not num == 0:
#         ll.extend([tokens["PAD_TOKEN"]] * num) # replace PAD_TOKEN
#     return ll
# def labelDictionary():
#     # Use dataset's character_classes
#     return len(labels), letter2index, index2letter
# tok = False
# if not tok:
#     tokens = {"PAD_TOKEN": 52}
# else:
#     tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
# num_tokens = len(tokens.keys())
# def setup_logging(args):
#     # os.makedirs("models", exist_ok=True)
#     os.makedirs(args.save_path, exist_ok=True)
#     os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
#     os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
# def save_images(images, path, args, **kwargs):
#     # print('image', images.shape)
#     grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
#     if args.latent == True:
#         im = torchvision.transforms.ToPILImage()(grid)
#         if args.color == False:
#             im = im.convert('L')
#         else:
#             im = im.convert('RGB')
#     else:
#         ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
#         im = Image.fromarray(ndarr)
#     im.save(path)
#     return im
# def crop_whitespace_width(img):
#     # tensor image to PIL
#     original_height = img.height
#     img_gray = np.array(img)
#     ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     coords = cv2.findNonZero(thresholded)
#     x, y, w, h = cv2.boundingRect(coords)
#     # rect = img.crop((x, 0, x + w, original_height))
#     rect = img.crop((x, y, x + w, y + h))
#     return np.array(rect)
# class AvgMeter:
#     def __init__(self, name="Metric"):
#         self.name = name
#         self.reset()
#     def reset(self):
#         self.avg, self.sum, self.count = [0] * 3
#     def update(self, val, count=1):
#         self.count += count
#         self.sum += val * count
#         self.avg = self.sum / self.count
#     def __repr__(self):
#         text = f"{self.name}: {self.avg:.4f}"
#         return text
# class EMA:
#     ''' EMA is used to stabilize the training process of diffusion models by
#     computing a moving average of the parameters, which can help to reduce the noise in the gradients
#     and improve the performance of the model.
#     '''
#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta
#         self.step = 0
#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)
#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new
#     def step_ema(self, ema_model, model, step_start_ema=2000):
#         if self.step < step_start_ema:
#             self.reset_parameters(ema_model, model)
#             self.step += 1
#             return
#         self.update_model_average(ema_model, model)
#         self.step += 1
#     def reset_parameters(self, ema_model, model):
#         ema_model.load_state_dict(model.state_dict())
# class Diffusion:
#     def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 1024), args=None):
#         self.noise_steps = noise_steps
#         self.beta_start = beta_start
#         self.beta_end = beta_end
#         self.beta = self.prepare_noise_schedule().to(args.device)
#         self.alpha = 1. - self.beta
#         self.alpha_hat = torch.cumprod(self.alpha, dim=0)
#         self.img_size = img_size
#         self.device = args.device
#     def prepare_noise_schedule(self):
#         return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
#     def sample_timesteps(self, n):
#         return torch.randint(low=1, high=self.noise_steps, size=(n,))
#     def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
#         model.eval()
#         tensor_list = []
#         with torch.no_grad():
#             pbar = tqdm(test_loader)
#             style_feat = []
#             for i, data in enumerate(pbar):
#                 images = data[0].to(args.device)
#                 transcr = data[1]
#                 s_id = torch.tensor([int(w) for w in data[3]]).to(args.device) # wid is data[3]
#                 style_images = data[7].to(args.device)
#                 cor_im = data[5].to(args.device)
#                 img_path = data[8]
#                 if args.model_name == 'wordstylist':
#                     # print('transcr', transcr)
#                     batch_word_embeddings = []
#                     for trans in transcr:
#                         word_embedding = label_padding(trans)
#                         # print('word_embedding', word_embedding)
#                         word_embedding = np.array(word_embedding, dtype="int64")
#                         word_embedding = torch.from_numpy(word_embedding).long()
#                         batch_word_embeddings.append(word_embedding)
#                     text_features = torch.stack(batch_word_embeddings).to(args.device)
#                 else:
#                     text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
#                 reshaped_images = style_images.reshape(-1, 3, 64, 1024)
#                 style_features = None
#                 if args.latent == True:
#                     x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
#                 else:
#                     x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
#                 # scheduler
#                 noise_scheduler.set_timesteps(50)
#                 for time in noise_scheduler.timesteps:
#                     t_item = time.item()
#                     t = (torch.ones(images.size(0)) * t_item).long().to(args.device)
#                     with torch.no_grad():
#                         noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
#                         prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
#                         x = prev_noisy_sample
#         model.train()
#         if args.latent==True:
#             latents = 1 / 0.18215 * x
#             image = vae.module.decode(latents).sample
#             image = (image / 2 + 0.5).clamp(0, 1)
#             image = image.cpu().permute(0, 2, 3, 1).numpy()
#             image = torch.from_numpy(image)
#             x = image.permute(0, 3, 1, 2)
#         else:
#             x = (x.clamp(-1, 1) + 1) / 2
#             x = (x * 255).type(torch.uint8)
#         return x
#     def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
#         model.eval()
#         tensor_list = []
#         with torch.no_grad():
#             style_images = None
#             text_features = x_text # [x_text]*n
#             # print('text features', text_features.shape)
#             text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
#             style_features = None
#             if args.latent == True:
#                 x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
#             else:
#                 x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
#             # scheduler
#             noise_scheduler.set_timesteps(50)
#             for time in noise_scheduler.timesteps:
#                 t_item = time.item()
#                 t = (torch.ones(n) * t_item).long().to(args.device)
#                 with torch.no_grad():
#                     noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
#                     prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
#                     x = prev_noisy_sample
#         model.train()
#         if args.latent==True:
#             latents = 1 / 0.18215 * x
#             image = vae.module.decode(latents).sample
#             image = (image / 2 + 0.5).clamp(0, 1)
#             image = image.cpu().permute(0, 2, 3, 1).numpy()
#             image = torch.from_numpy(image)
#             x = image.permute(0, 3, 1, 2)
#         else:
#             x = (x.clamp(-1, 1) + 1) / 2
#             x = (x * 255).type(torch.uint8)
#         return x
# def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
#     model.train()
#     loss_meter = AvgMeter()
#     print('Training started....')
#     for epoch in range(args.epochs):
#         print('Epoch:', epoch)
#         pbar = tqdm(loader)
#         style_feat = []
#         for i, data in enumerate(pbar):
#             images = data[0].to(args.device)
#             transcr = data[1]
#             s_id = torch.tensor([int(w) for w in data[3]]).to(args.device) # wid is data[3]
#             style_images = data[7].to(args.device) # s_imgs is data[7]
#             if args.model_name == 'wordstylist':
#                 batch_word_embeddings = []
#                 for trans in transcr:
#                     word_embedding = label_padding(trans, num_tokens)
#                     word_embedding = np.array(word_embedding, dtype="int64")
#                     word_embedding = torch.from_numpy(word_embedding).long()
#                     batch_word_embeddings.append(word_embedding)
#                 text_features = torch.stack(batch_word_embeddings)
#             else:
#                 text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
#             style_features = None
#             if args.latent == True:
#                 images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
#                 images = images * 0.18215
#             latents = images
#             noise = torch.randn(images.shape).to(images.device)
#             # Sample a random timestep for each image
#             num_train_timesteps = diffusion.noise_steps
#             timesteps = torch.randint(
#                 0, num_train_timesteps, (images.shape[0],), device=images.device
#             ).long()
#             # Add noise to the clean images according to the noise magnitude
#             # at each timestep (this is the forward diffusion process)
#             noisy_images = noise_scheduler.add_noise(
#                 images, noise, timesteps
#             )
#             x_t = noisy_images
#             t = timesteps
#             if np.random.random() < 0.1:
#                 labels = None
#             predicted_noise = model(x_t, timesteps=t, context=text_features, y=s_id, style_extractor=style_features)
#             loss = mse_loss(noise, predicted_noise)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             ema.step_ema(ema_model, model)
#             count = images.size(0)
#             loss_meter.update(loss.item(), count)
#             pbar.set_postfix(MSE=loss_meter.avg)
#             if lr_scheduler is not None:
#                 lr_scheduler.step()
#         if epoch % 10 == 0:
#             batch_size = test_loader.batch_size
#             labels = torch.arange(batch_size).long().to(args.device) % num_classes
#             n = batch_size
#             if args.sampling_word == True:
#                 # generates the word "text" in 16 different styles
#                 words = ['text']
#                 for x_text in words:
#                     ema_sampled_images = diffusion.sample(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)
#                     epoch_n = epoch
#                     sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"), args)
#             else:
#                 # generates a batch of words
#                 ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=n, x_text=None, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
#                 epoch_n = epoch
#                 sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
#             if args.wandb_log==True:
#                 wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
#                 wandb.log({f"Sampled images": wandb_sampled_ema})
#             # Generate one Urdu sentence in one style
#             urdu_text = 'کون سوچ سکتا تھا کہ ہندوستان اکثریت اورانگریزحکمرانوں کی مشترکہ'
#             s = 0 # fixed style
#             labels = torch.tensor([s]).long().to(args.device)
#             sample_image = diffusion.sampling(model=ema_model, vae=vae, n=1, x_text=urdu_text, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
#             save_images(sample_image, os.path.join(args.save_path, 'images', f'urdu_sample_epoch_{epoch}_style_{s}.jpg'), args)
#         if epoch==490:
#             torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
#             torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
#             torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))
# def main():
#     '''Main function'''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=1000)
#     parser.add_argument('--batch_size', type=int, default=320)
#     parser.add_argument('--num_workers', type=int, default=4)
#     parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
#     parser.add_argument('--level', type=str, default='line', help='word, line') # Changed to line for sentences
#     parser.add_argument('--img_size', type=int, default=(64, 1024))
#     # parser.add_argument('--img_size', type=int, default=(128, 512))
#     parser.add_argument('--dataset', type=str, default='unhd', help='iam, gnhk, unhd')
#     # UNET parameters
#     parser.add_argument('--channels', type=int, default=4)
#     parser.add_argument('--emb_dim', type=int, default=320)
#     parser.add_argument('--num_heads', type=int, default=4)
#     parser.add_argument('--num_res_blocks', type=int, default=1)
#     parser.add_argument('--save_path', type=str, default='./diffusionpen_unhd_model_path')
#     parser.add_argument('--device', type=str, default='cuda:0')
#     parser.add_argument('--wandb_log', type=bool, default=False)
#     parser.add_argument('--color', type=bool, default=True)
#     parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
#     parser.add_argument('--latent', type=bool, default=True)
#     parser.add_argument('--img_feat', type=bool, default=False)
#     parser.add_argument('--interpolation', type=bool, default=False)
#     parser.add_argument('--dataparallel', type=bool, default=False)
#     parser.add_argument('--load_check', type=bool, default=False)
#     parser.add_argument('--sampling_word', type=bool, default=False)
#     parser.add_argument('--mix_rate', type=float, default=None)
#     parser.add_argument('--style_path', type=str, default='./style_models/unhd_style_diffusionpen.pth')
#     parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
#     parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
#     parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
#     args = parser.parse_args()
#     print('torch version', torch.__version__)
#     if args.wandb_log==True:
#         runs = wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)
#         wandb.config.update(args)
#     # create save directories
#     setup_logging(args)
#     ############################ DATASET ############################
#     transform = transforms.Compose([
#         # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
#         transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # transforms.Normalize((0.5,), (0.5,)),
#     ])
#     unhd_folder = args.dataset
#     myDataset = UNHDDataset
#     if args.level == 'line':
#         train_data = myDataset(unhd_folder, 'train', 'line', fixed_size=(1 * 64, 1024), transforms=transform, args=args)
#     test_data = myDataset(unhd_folder, 'test', 'line', fixed_size=(1 * 64, 1024), transforms=transform, args=args)
#     print('train data', len(train_data))
#     test_size = args.batch_size
#     rest = len(train_data) - test_size
#     test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
#     with open('./writers_dict.json', 'r') as f:
#         wr_dict = json.load(f)
#     style_classes = len(wr_dict)
#     character_classes = train_data.character_classes
#     vocab_size = len(character_classes) + num_tokens
#     print('num of character classes', vocab_size)
#     train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
#     ######################### MODEL #######################################
#     if args.model_name == 'wordstylist':
#         vocab_size = len(character_classes) + 2
#         print('vocab size', vocab_size)
#     else:
#         vocab_size = len(character_classes)
#     print('Vocab size: ', vocab_size)
#     if args.dataparallel==True:
#         device_ids = [3,4]
#         print('using dataparallel with device:', device_ids)
#     else:
#         idx = int(''.join(filter(str.isdigit, args.device)))
#         device_ids = [idx]
#     if args.model_name == 'diffusionpen':
#         tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
#         text_encoder = CanineModel.from_pretrained("google/canine-c")
#         text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
#         text_encoder = text_encoder.to(args.device)
#     else:
#         tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
#         text_encoder = None
#     if args.unet=='unet_latent':
#         unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)#.to(args.device)
#         unet = DataParallel(unet, device_ids=device_ids)
#         unet = unet.to(args.device)
#         # print('unet parameters')
#         # print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))
#     optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
#     lr_scheduler = None
#     mse_loss = nn.MSELoss()
#     diffusion = Diffusion(img_size=args.img_size, args=args)
#     ema = EMA(0.995)
#     ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
#     # load from last checkpoint
#     if args.load_check==True:
#         unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
#         optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
#         ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
#         print('Loaded models and optimizer')
#     if args.latent==True:
#         print('VAE is true')
#         vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
#         vae = DataParallel(vae, device_ids=device_ids)
#         vae = vae.to(args.device)
#         # Freeze vae and text_encoder
#         vae.requires_grad_(False)
#     else:
#         vae = None
#     # add DDIM scheduler from huggingface
#     ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
#     #### STYLE ####
#     feature_extractor = None
#     if args.train_mode == 'train':
#         train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
#     elif args.train_mode == 'sampling':
#         print('Sampling started....')
#         unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
#         print('unet loaded')
#         ema = EMA(0.995)
#         ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
#         ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
#         ema_model.eval()
#         if args.sampling_mode == 'single_sampling':
#             x_text = ['کون', 'سوچ'] # Urdu text examples
#             for x_text in x_text:
#                 print('Word:', x_text)
#                 s = random.randint(0, style_classes - 1) # index for style class
#                 print('style', s)
#                 labels = torch.tensor([s]).long().to(args.device)
#                 ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
#                 save_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)
#         elif args.sampling_mode == 'paragraph':
#             print('Sampling paragraph')
#             #make the code to generate lines
#             lines = 'کون سوچ سکتا تھا کہ ہندوستان اکثریت اورانگریزحکمرانوں کی مشترکہ مخالفت کےباوجود برصغیرکی ملت اسلامیہ دین اسلام ہے اور اسی نظریہ پر اس ملک میں بسنے والے مختلف عناصر کا اتحاد ہےاورپاکستان کی بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔' # Example Urdu text
#             fakes= []
#             gap = np.ones((64, 16)) * 255
#             max_line_width = 900
#             total_char_count = 0
#             avg_char_width = 0
#             current_line_width = 0
#             words = lines.strip().split(' ')
#             longest_word_length = max(len(word) for word in words)
#             #print('longest_word_length', longest_word_length)
#             s = random.randint(0, style_classes - 1)#.long().to(args.device)
#             print('Style:', s)
#             words = words[::-1] # Reverse for RTL
#             for word in words:
#                 print('Word:', word)
#                 labels = torch.tensor([s]).long().to(args.device)
#                 ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
#                 image = ema_sampled_images.squeeze(0)
#                 im = torchvision.transforms.ToPILImage()(image)
#                 im = im.convert("L")
#                 im = crop_whitespace_width(im)
#                 im = Image.fromarray(im)
#                 if len(word) == longest_word_length:
#                     max_word_length_width = im.width
#                 # Calculate aspect ratio
#                 aspect_ratio = im.width / im.height
#                 im = np.array(im)
#                 fakes.append(im)
#             # Calculate the scaling factor based on the longest word
#             #find the average character width of the max length word
#             avg_char_width = max_word_length_width / longest_word_length
#             print('avg_char_width', avg_char_width)
#             #scaling_factor = avg_char_width / (32 * aspect_ratio) # Aspect ratio of an average character
#             # Scale and pad each word
#             scaled_padded_words = []
#             max_height = 64 # Defined max height for all images
#             punctuation = '.,?!:;،۔؟' # Added Urdu punctuation
#             for word, img in zip(words, fakes):
#                 img_pil = Image.fromarray(img)
#                 as_ratio = img_pil.width / img_pil.height
#                 #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
#                 scaled_width = int(avg_char_width * len(word))
#                 scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
#                 print(f'Word {word} - scaled_img {scaled_img.size}')
#                 # Padding
#                 #if word is in punctuation:
#                 if any(char in punctuation for char in word):
#                     #rescale to height 10
#                     w_punc = scaled_img.width
#                     h_punc = scaled_img.height
#                     as_ratio_punct = w_punc / h_punc
#                     if '.' in word or '۔' in word:
#                         scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
#                     else:
#                         scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
#                     #pad on top and leave the image in the bottom
#                     padding_bottom = 10
#                     padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
#                     # No padding at the bottom
#                     # Apply padding
#                     padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
#                 else:
#                     if scaled_img.height < max_height:
#                         padding = (max_height - scaled_img.height) // 2
#                         #print(f'Word {word} - padding: {padding}')
#                         padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
#                     else:
#                         #resize to max height while maintaining aspect ratio
#                         #ar = scaled_img.width / scaled_img.height
#                         scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
#                         padding = (max_height - scaled_img.height) // 2
#                         #print(f'Word {word} - padding: {padding}')
#                         padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
#                     #padded_img = np.array(scaled_img)
#                     #print('padded_img', padded_img.shape)
#                 scaled_padded_words.append(padded_img)
#             # Create a gap array (white space)
#             height = 64 # Fixed height for all images
#             gap = np.ones((height, 16), dtype=np.uint8) * 255 # White gap
#             # Concatenate images with gaps
#             sentence_img = gap # Start with a gap
#             lines = []
#             line_img = gap
#             ''' sentence_img = gap # Start with a gap
#             for img in scaled_padded_words:
#                 #print('img', img.shape)
#                 sentence_img = np.concatenate((sentence_img, img, gap), axis=1) '''
#             for img in scaled_padded_words:
#                 img_width = img.shape[1] + gap.shape[1]
#                 if current_line_width + img_width < max_line_width:
#                     # Add the image to the current line
#                     if line_img.shape[0] == 0:
#                         line_img = np.ones((height, 0), dtype=np.uint8) * 255 # Start a new line
#                     line_img = np.concatenate((line_img, img, gap), axis=1)
#                     current_line_width += img_width #+ gap.shape[1]
#                     #print('current_line_width if', current_line_width)
#                     # Check if adding this image exceeds the max line width
#                 else:
#                     # Pad the current line with white space to max_line_width
#                     remaining_width = max_line_width - current_line_width
#                     line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
#                     lines.append(line_img)
#                     # Start a new line with the current word
#                     line_img = np.concatenate((gap, img, gap), axis=1)
#                     current_line_width = img_width #+ 2 * gap.shape[1]
#                     #print('current_line_width else', current_line_width)
#             # Add the last line to the lines list if current_line_width > 0
#             if current_line_width > 0:
#                 # Pad the last line to max_line_width
#                 remaining_width = max_line_width - current_line_width
#                 line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
#                 lines.append(line_img)
#             # # Concatenate all lines to form a paragraph, pad them if necessary
#             # max_height = max([line.shape[0] for line in lines])
#             # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
#             # for line in lines:
#             # if line.shape[0] < max_height:
#             # padding = (max_height - line.shape[0]) // 2
#             # line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)
#             # #print the shapes
#             # print('line shape', line.shape)
#             #print('paragraph shape', paragraph_img.shape)
#             paragraph_img = np.concatenate(lines, axis=0)
#             paragraph_image = Image.fromarray(paragraph_img)
#             paragraph_image = paragraph_image.convert("L")
#             paragraph_image.save(f'paragraph_style_{s}.png')
#         elif args.sampling_mode == 'multi_sentence':
#             print('Sampling multi sentences')
#             sentences = [
#                 'مخالفت کےباوجود برصغیرکی ملت اسلامیہ دین اسلام ہے اور اسی نظریہ',
#                 'بقاء اسی نظریہ حیات کے فروغ پر منحصر ہے۔',
#                 'لیکن بدقسمتی سےپاکستان بننے کے بعد ہی اس کے اندر ایسے دشمن',
#                 'میں مصروف ہوگیا بغیراسکا اہتمام کۓ کہ جسکا وہ پھل کھا رہا ہے'
#             ]
#             s = random.randint(0, style_classes - 1) # One random style for all
#             print('Style:', s)
#             labels = torch.tensor([s]).long().to(args.device)
#             for idx, sentence in enumerate(sentences):
#                 print('Sentence:', sentence)
#                 ema_sampled_images = diffusion.sampling(ema_model, vae, n=1, x_text=sentence, labels=labels, args=args, style_extractor=None, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
#                 save_images(ema_sampled_images, os.path.join(f'./image_samples/', f'sentence_{idx+1}_style_{s}.png'), args)
# if __name__ == "__main__":
#     main()



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
from unet import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from utils.unhd_dataset import UNHDDataset # Import UNHD dataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 # +2 # <GO>+groundtruth+<END>
IMG_WIDTH = 1024
IMG_HEIGHT = 64
# For Urdu, character classes are auto-computed in dataset
# No fixed c_classes, use dataset.character_classes
### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    # ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num) # replace PAD_TOKEN
    return ll
def labelDictionary():
    # Use dataset's character_classes
    return len(labels), letter2index, index2letter
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
def setup_logging(args):
    # os.makedirs("models", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
def save_images(images, path, args, **kwargs):
    # print('image', images.shape)
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im
def crop_whitespace_width(img):
    # tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    # rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()
    def reset(self):
        self.avg, self.sum, self.count = [0] * 3
    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count
    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text
class EMA:
    ''' EMA is used to stabilize the training process of diffusion models by
    computing a moving average of the parameters, which can help to reduce the noise in the gradients
    and improve the performance of the model.
    '''
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
        tensor_list = []
        with torch.no_grad():
            pbar = tqdm(test_loader)
            style_feat = []
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                s_id = torch.tensor([int(w) for w in data[3]]).to(args.device) # wid is data[3]
                style_images = data[7].to(args.device)
                cor_im = data[5].to(args.device)
                img_path = data[8]
                if args.model_name == 'wordstylist':
                    # print('transcr', transcr)
                    batch_word_embeddings = []
                    for trans in transcr:
                        word_embedding = label_padding(trans)
                        # print('word_embedding', word_embedding)
                        word_embedding = np.array(word_embedding, dtype="int64")
                        word_embedding = torch.from_numpy(word_embedding).long()
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                reshaped_images = style_images.reshape(-1, 3, 64, 1024)
                style_features = None
                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
                # scheduler
                noise_scheduler.set_timesteps(100)  # Increased for better quality
                for time in noise_scheduler.timesteps:
                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)
                    with torch.no_grad():
                        # Classifier-free guidance
                        uncond_features = tokenizer([""] * images.size(0), padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                        uncond_residual = model(x, t, uncond_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                        noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                        noisy_residual = uncond_residual + cfg_scale * (noisy_residual - uncond_residual)
                        prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                        x = prev_noisy_sample
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x
    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        tensor_list = []
        with torch.no_grad():
            style_images = None
            text_features = x_text # [x_text]*n
            # print('text features', text_features.shape)
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
            style_features = None
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            # scheduler
            noise_scheduler.set_timesteps(100)  # Increased for better quality
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)
                with torch.no_grad():
                    # Classifier-free guidance
                    uncond_features = tokenizer([""] * n, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                    uncond_residual = model(x, t, uncond_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    noisy_residual = uncond_residual + cfg_scale * (noisy_residual - uncond_residual)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x
def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        style_feat = []
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = torch.tensor([int(w) for w in data[3]]).to(args.device) # wid is data[3]
            style_images = data[7].to(args.device) # s_imgs is data[7]
            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long()
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
            style_features = None
            if args.latent == True:
                images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
            latents = images
            noise = torch.randn(images.shape).to(images.device)
            # Sample a random timestep for each image
            num_train_timesteps = diffusion.noise_steps
            timesteps = torch.randint(
                0, num_train_timesteps, (images.shape[0],), device=images.device
            ).long()
            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )
            x_t = noisy_images
            t = timesteps
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, timesteps=t, context=text_features, y=s_id, style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)
            if lr_scheduler is not None:
                lr_scheduler.step()
        if epoch % 10 == 0:
            batch_size = test_loader.batch_size
            labels = torch.arange(batch_size).long().to(args.device) % num_classes
            n = batch_size
            if args.sampling_word == True:
                # generates the word "text" in 16 different styles
                words = ['text']
                for x_text in words:
                    ema_sampled_images = diffusion.sample(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)
                    epoch_n = epoch
                    sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"), args)
            else:
                # generates a batch of words
                ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=n, x_text=None, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                epoch_n = epoch
                sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
            if args.wandb_log==True:
                wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"{x_text}_{epoch}")
                wandb.log({f"Sampled images": wandb_sampled_ema})
            # Generate one Urdu sentence in one style
            urdu_text = 'کون سوچ سکتا تھا کہ ہندوستان اکثریت اورانگریزحکمرانوں کی مشترکہ'
            s = 0 # fixed style
            labels = torch.tensor([s]).long().to(args.device)
            sample_image = diffusion.sampling(model=ema_model, vae=vae, n=1, x_text=urdu_text, labels=labels, args=args, style_extractor=None, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
            save_images(sample_image, os.path.join(args.save_path, 'images', f'urdu_sample_epoch_{epoch}_style_{s}.jpg'), args)
        if epoch==990:
            torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))
def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='line', help='word, line') # Changed to line for sentences
    parser.add_argument('--img_size', type=int, default=(64, 1024))
    # parser.add_argument('--img_size', type=int, default=(128, 512))
    parser.add_argument('--dataset', type=str, default='unhd', help='iam, gnhk, unhd')
    # UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_unhd_model_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/unhd_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    args = parser.parse_args()
    print('torch version', torch.__version__)
    if args.wandb_log==True:
        runs = wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)
        wandb.config.update(args)
    # create save directories
    setup_logging(args)
    ############################ DATASET ############################
    transform = transforms.Compose([
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # transforms.Normalize((0.5,), (0.5,)),
    ])
    unhd_folder = args.dataset
    myDataset = UNHDDataset
    if args.level == 'line':
        train_data = myDataset(unhd_folder, 'train', 'line', fixed_size=(1 * 64, 1024), transforms=transform, args=args)
    test_data = myDataset(unhd_folder, 'test', 'line', fixed_size=(1 * 64, 1024), transforms=transform, args=args)
    print('train data', len(train_data))
    test_size = args.batch_size
    rest = len(train_data) - test_size
    test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
    with open('./writers_dict.json', 'r') as f:
        wr_dict = json.load(f)
    style_classes = len(wr_dict)
    character_classes = train_data.character_classes
    vocab_size = len(character_classes) + num_tokens
    print('num of character classes', vocab_size)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    ######################### MODEL #######################################
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
        print('vocab size', vocab_size)
    else:
        vocab_size = len(character_classes)
    print('Vocab size: ', vocab_size)
    if args.dataparallel==True:
        device_ids = [3,4]
        print('using dataparallel with device:', device_ids)
    else:
        idx = int(''.join(filter(str.isdigit, args.device)))
        device_ids = [idx]
    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)#.to(args.device)
        unet = DataParallel(unet, device_ids=device_ids)
        unet = unet.to(args.device)
        # print('unet parameters')
        # print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    # load from last checkpoint
    if args.load_check==True:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None
    # add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    #### STYLE ####
    feature_extractor = None
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
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
                    if line_img.shape[0] == 0:
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