""" Word Generation Dataset Loader for DiffusionPen
Handles word-level Urdu handwritten images for training
"""
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFile, ImageFilter
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from torchvision import transforms

# CRITICAL: Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_resize_PIL(img, height=None, width=None):
    """Resize PIL image maintaining aspect ratio"""
    if height is None and width is None:
        return img
    orig_width, orig_height = img.size
    if orig_height == 0 or orig_width == 0:
        return img
    if height is not None and width is None:
        ratio = height / orig_height
        width = max(1, int(orig_width * ratio))
    elif width is not None and height is None:
        ratio = width / orig_width
        height = max(1, int(orig_height * ratio))
    height = max(1, height)
    width = max(1, width)
    try:
        return img.resize((width, height), Image.LANCZOS)
    except Exception as e:
        print(f"Warning: Error resizing image: {e}")
        return img

def centered_PIL(img, target_size, border_value=255):
    """Center image on a canvas of target_size"""
    target_h, target_w = target_size
    if img.mode == 'L':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    canvas = Image.new('RGB', (target_w, target_h), color=(border_value, border_value, border_value))
    img_w, img_h = img.size
    paste_x = max(0, (target_w - img_w) // 2)
    paste_y = max(0, (target_h - img_h) // 2)
    try:
        canvas.paste(img, (paste_x, paste_y))
    except Exception as e:
        print(f"Warning: Error pasting image: {e}")
    return canvas

class WordGenerationDataset(Dataset):
    def __init__(self, image_folder, gt_folder, subset='train', fixed_size=(64, 256), 
                 tokenizer=None, text_encoder=None, feat_extractor=None, transforms=None, args=None):
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.subset = subset
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.args = args
        self.setname = 'WordGeneration'
        
        self.data = self.main_loader()
        
        self.wclasses = 1
        self.writer_ids = [0]
        print(f'Number of styles: {self.wclasses}')
        
        res = set()
        self.max_transcr_len = 0
        for _, transcr, _, _ in tqdm(self.data, desc="Building character classes"):
            res.update(list(transcr))
            self.max_transcr_len = max(self.max_transcr_len, len(transcr))
        
        res = sorted(list(res))
        res.append(' ')
        self.character_classes = res
        print(f'Character classes: {len(res)} different characters')
        print(f'Max transcription length: {self.max_transcr_len}')

        # Word-level uses a single dummy style (wid=0 for every sample), so the
        # "positive_samples" pool is identical for every item -- compute it once
        # here instead of rescanning the whole dataset inside __getitem__.
        self._samples_by_wid = {}
        for p in self.data:
            self._samples_by_wid.setdefault(p[2], []).append(p)

    def load_and_verify_image(self, img_path):
        try:
            img = Image.open(img_path)
            img.verify()
            img = Image.open(img_path)
            img.load()
            
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            if img.width == 0 or img.height == 0:
                return None
            
            return img
            
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            return None

    def main_loader(self):
        """Load all word images and their ground truth texts"""
        print(f'Loading data from:')
        print(f'  Images: {self.image_folder}')
        print(f'  Ground truth: {self.gt_folder}')
        
        data_temp = []
        failed_count = 0
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend([f for f in os.listdir(self.image_folder) if f.endswith(ext)])
        
        text_files = [f for f in os.listdir(self.gt_folder) if f.endswith('.txt')]
        
        print(f'Found {len(image_files)} image files')
        print(f'Found {len(text_files)} text files')
        
        text_file_map = {}
        for txt_file in text_files:
            base_name = os.path.splitext(txt_file)[0]
            text_file_map[base_name] = txt_file
        
        print(f'Created text file mapping with {len(text_file_map)} entries')
        
        def natural_sort_key(filename):
            base = os.path.splitext(filename)[0]
            try:
                return int(base)
            except ValueError:
                return base
        
        image_files = sorted(image_files, key=natural_sort_key)
        
        print(f'First 5 image files after sorting: {image_files[:5]}')
        
        for img_file in tqdm(image_files, desc=f"Loading {self.subset} data"):
            base_name = os.path.splitext(img_file)[0]
            
            img_path = os.path.join(self.image_folder, img_file)
            
            if base_name not in text_file_map:
                if failed_count < 10:
                    print(f"Warning: Text file not found for {img_file} (looking for {base_name}.txt)")
                failed_count += 1
                continue
            
            txt_file = text_file_map[base_name]
            txt_path = os.path.join(self.gt_folder, txt_file)
            
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcr = f.read().strip()
                
                if not transcr:
                    if failed_count < 10:
                        print(f"Warning: Empty transcription for {img_file}")
                    failed_count += 1
                    continue
                
                style_id = 0
                
                data_temp.append((img_path, transcr, style_id, img_path))
                
            except Exception as e:
                if failed_count < 10:
                    print(f"Error reading {txt_path}: {e}")
                failed_count += 1
                continue
        
        print(f"\nTotal data collected: {len(data_temp)} samples")
        print(f"Failed/skipped: {failed_count} samples")
        
        if len(data_temp) == 0:
            print("\n" + "="*80)
            print("ERROR: No valid samples found!")
            print("="*80)
            print("Debugging info:")
            print(f"  Total image files found: {len(image_files)}")
            print(f"  Total text files found: {len(text_files)}")
            print(f"  Text file map size: {len(text_file_map)}")
            print(f"\n  Sample image files (first 5): {image_files[:5] if image_files else 'None'}")
            print(f"  Sample text file map keys (first 5): {list(text_file_map.keys())[:5] if text_file_map else 'None'}")
            print("\nPlease check that:")
            print("  1. Image and text files have matching names (e.g., 1.jpg <-> 1.txt)")
            print("  2. Text files are not empty")
            print("  3. Files are readable and properly encoded (UTF-8)")
            print("="*80)
            raise ValueError("No valid samples found in dataset")
        
        print(f"Loaded {len(data_temp)} {self.subset} samples (failed: {failed_count})")
        return data_temp

    def __len__(self):
        return len(self.data)

    def _process_image(self, img, fheight, fwidth):
        from PIL import Image, ImageFilter, ImageOps
        import numpy as np
        
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
        try:
            img.load()
        except Exception:
            return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
        if img.mode == 'L':
            img = img.convert('RGB')
        elif img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            try:
                background.paste(img, mask=img.split()[3])
            except Exception:
                background.paste(img)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        if img.width == 0 or img.height == 0:
            return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=10))
        img = ImageOps.autocontrast(img, cutoff=2)
        
        orig_w, orig_h = img.size
        
        max_h = max(4, fheight - 4)
        max_w = max(8, fwidth - 8)
        
        scale_h = max_h / float(orig_h)
        scale_w = max_w / float(orig_w)
        base_scale = min(scale_h, scale_w)
        
        scale = base_scale
        
        target_h = max(4, int(orig_h * scale))
        
        img = image_resize_PIL(img, height=target_h, width=None)
        
        img = centered_PIL(img, (fheight, fwidth), border_value=255)
        
        return img

    def __getitem__(self, index):
        try:
            img_path, transcr, wid, _ = self.data[index]
            img = self.load_and_verify_image(img_path)
            
            if img is None:
                raise Exception("Failed to load image")
            
            positive_samples = self._samples_by_wid.get(wid, [])
            
            if len(positive_samples) == 0:
                positive_samples = [self.data[index]]
            
            positive_path = random.choice(positive_samples)[0]
            positive = self.load_and_verify_image(positive_path)
            if positive is None:
                positive = Image.new('RGB', (self.fixed_size[1], self.fixed_size[0]), (255, 255, 255))
            
            negative = positive
            
            k = min(5, len(positive_samples))
            if k > 0:
                random_samples = random.sample(positive_samples, k=k)
                style_paths = [i[0] for i in random_samples]
                while len(style_paths) < 5:
                    style_paths.append(style_paths[0])
            else:
                style_paths = [img_path] * 5
            
            fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
            
            img = self._process_image(img, fheight, fwidth)
            positive = self._process_image(positive, fheight, fwidth)
            negative = self._process_image(negative, fheight, fwidth)
            
            st_imgs = []
            for s_path in style_paths:
                s_img = self.load_and_verify_image(s_path)
                if s_img is None:
                    s_img = Image.new('RGB', (fwidth, fheight), (255, 255, 255))
                s_img = self._process_image(s_img, fheight, fwidth)
                if self.transforms is not None:
                    s_img_tensor = self.transforms(s_img)
                else:
                    s_img_tensor = transforms.ToTensor()(s_img)
                st_imgs.append(s_img_tensor)
            
            s_imgs = torch.stack(st_imgs)
            
            if self.transforms is not None:
                img = self.transforms(img)
                positive = self.transforms(positive)
                negative = self.transforms(negative)
            else:
                img = transforms.ToTensor()(img)
                positive = transforms.ToTensor()(positive)
                negative = transforms.ToTensor()(negative)
            
            char_tokens = [self.character_classes.index(c) if c in self.character_classes else 0 for c in transcr]
            pad_token = len(self.character_classes) - 1
            padding_length = 95 - len(char_tokens)
            if padding_length > 0:
                char_tokens.extend([pad_token] * padding_length)
            else:
                char_tokens = char_tokens[:95]
            char_tokens = torch.tensor(char_tokens, dtype=torch.long)
            
            return img, transcr, char_tokens, wid, positive, negative, self.character_classes, s_imgs, img_path
            
        except Exception as e:
            print(f"Error in __getitem__ at index {index}: {e}")
            fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
            dummy_img = Image.new('RGB', (fwidth, fheight), (255, 255, 255))
            if self.transforms is not None:
                dummy_tensor = self.transforms(dummy_img)
            else:
                dummy_tensor = transforms.ToTensor()(dummy_img)
            
            dummy_transcr = " "
            dummy_wid = 0
            dummy_tokens = torch.zeros(95, dtype=torch.long)
            dummy_style = torch.stack([dummy_tensor] * 5)
            
            return dummy_tensor, dummy_transcr, dummy_tokens, dummy_wid, dummy_tensor, dummy_tensor, [' '], dummy_style, ""

    def collate_fn(self, batch):
        img, transcr, char_tokens, wid, positive, negative, cla, s_imgs, img_path = zip(*batch)
        
        images_batch = torch.stack(img)
        char_tokens_batch = torch.stack(char_tokens)
        images_pos = torch.stack(positive)
        images_neg = torch.stack(negative)
        s_imgs_batch = torch.stack(s_imgs)
        
        return images_batch, transcr, char_tokens_batch, wid, images_pos, images_neg, cla[0], s_imgs_batch, img_path