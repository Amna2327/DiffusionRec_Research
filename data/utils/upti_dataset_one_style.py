""" Fixed UPTI Dataset Loader for DiffusionPen
Handles grayscale images and truncated image errors properly
"""
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import random
from torchvision import transforms
# CRITICAL: Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True
def image_resize_PIL(img, height=None, width=None):
    """Resize PIL image maintaining aspect ratio"""
    if height is None and width is None:
        return img
    orig_width, orig_height = img.size
    # Handle zero dimensions
    if orig_height == 0 or orig_width == 0:
        return img
    if height is not None and width is None:
        ratio = height / orig_height
        width = max(1, int(orig_width * ratio))
    elif width is not None and height is None:
        ratio = width / orig_width
        height = max(1, int(orig_height * ratio))
    # Ensure dimensions are at least 1x1
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
    # Convert grayscale to RGB if needed
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
class UPTIDataset(Dataset):
    def __init__(self, basefolder, gt_folder, subset, segmentation_level, fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=None, args=None):
        self.basefolder = basefolder
        self.gt_folder = gt_folder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = (64, 1024)
        self.transforms = transforms
        self.args = args
        self.setname = 'UPTI'
        # Load data (paths only)
        self.data = self.main_loader(segmentation_level)
        # Get style info (fonts as styles)
        self.wclasses = len(self.writer_ids)
        print(f'Number of styles (fonts): {self.wclasses}')
        # Character classes
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
    def load_and_verify_image(self, img_path):
        """Load image with proper error handling and conversion"""
        try:
            # Open image
            img = Image.open(img_path)
            # Verify the image can be loaded
            img.verify()
            # Reopen after verify (verify closes the file)
            img = Image.open(img_path)
            # Force load the image data
            img.load()
            # Convert grayscale to RGB
            if img.mode == 'L':
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                # Handle transparency by creating white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            # Check valid dimensions
            if img.width == 0 or img.height == 0:
                return None
            return img
        except Exception as e:
            print(f'Error loading image {img_path}: {e}')
            return None
    def main_loader(self, segmentation_level):
        if segmentation_level != 'line':
            raise ValueError("UPTI supports 'line' level only")
        print('image folder', self.basefolder)
        print('gt folder', self.gt_folder)
        # Gather file info
        data_temp = []
        fonts = set()
        for line_dir in tqdm(os.listdir(self.basefolder), desc="Loading data"):
            if not line_dir.isdigit():
                continue
            line_path = os.path.join(self.basefolder, line_dir)
            if not os.path.isdir(line_path):
                continue
            gt_path = os.path.join(self.gt_folder, line_dir + '.txt')
            if not os.path.exists(gt_path):
                continue
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    transcr = f.read().strip()
            except Exception as e:
                print(f"Error reading {gt_path}: {e}")
                continue
            # NEW: Select only one style (font) - choose the first unique font found
            selected_font = None
            for font_dir in os.listdir(line_path):
                if selected_font is None:
                    selected_font = font_dir  # Pick the first font as the only style
                if font_dir != selected_font:
                    continue  # Skip other fonts
                font_path = os.path.join(line_path, font_dir)
                if not os.path.isdir(font_path):
                    continue
                fonts.add(font_dir)
                for deg_dir in os.listdir(font_path):
                    deg_path = os.path.join(font_path, deg_dir)
                    if not os.path.isdir(deg_path):
                        continue
                    img_name = line_dir + '.png'
                    img_path = os.path.join(deg_path, img_name)
                    if os.path.exists(img_path):
                        data_temp.append((img_path, transcr, font_dir))  # temp with font str
                # Break after processing selected font
                break
        # Cap at 100000 samples
        if len(data_temp) > 100000:
            data_temp = random.sample(data_temp, 100000)
        # Assign style IDs (will be only 1 since one font)
        unique_fonts = sorted(list(fonts))
        wr_dict = {font: idx for idx, font in enumerate(unique_fonts)}
        self.writer_ids = unique_fonts
        self.wr_dict = wr_dict
        # Update data with IDs
        data = [(img_path, transcr, wr_dict[font], img_path) for img_path, transcr, font in data_temp]
        print(f"Found {len(data)} valid samples (subset with one style)")
        return data
    def __len__(self):
        return len(self.data)
    def _process_image(self, img, fheight, fwidth):
        """
        Process image to fixed size with:
        - aspect-ratio preserving resize,
        - improved contrast & sharpening,
        - clean padding.
        This keeps the handwriting natural but clearer for the model.
        """
        from PIL import Image, ImageFilter, ImageOps
        import numpy as np
        # Ensure img is a PIL Image
        if not isinstance(img, Image.Image):
            try:
                img = Image.fromarray(img)
            except Exception:
                return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        # Verify load
        try:
            img.load()
        except Exception:
            return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        # Convert to RGB if needed
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
        # Safety check
        if img.width == 0 or img.height == 0:
            return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        # ---------------------------------------------------
        # 1. Soft Sharpening (best for clear handwriting)
        # ---------------------------------------------------
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=10))
        img = ImageOps.autocontrast(img, cutoff=2)
        # ---------------------------------------------------
        # 2. Aspect-ratio preserving resize (fills canvas)
        # ---------------------------------------------------
        orig_w, orig_h = img.size
        # Use almost entire canvas (tiny margins)
        max_h = max(4, fheight - 4)
        max_w = max(8, fwidth - 8)
        scale_h = max_h / float(orig_h)
        scale_w = max_w / float(orig_w)
        base_scale = min(scale_h, scale_w)
        # No jitter — preserve handwriting geometry
        scale = base_scale
        target_h = max(4, int(orig_h * scale))
        # Only set height → preserves aspect ratio
        img = image_resize_PIL(img, height=target_h, width=None)
        # ---------------------------------------------------
        # 3. Centered padding on final canvas
        # ---------------------------------------------------
        img = centered_PIL(img, (fheight, fwidth), border_value=255)
        return img
    def __getitem__(self, index):
        try:
            img_path, transcr, wid, _ = self.data[index]
            img = self.load_and_verify_image(img_path)
            if img is None:
                raise Exception("Failed to load image")
            # Get positive and negative samples (based on style/font ID)
            positive_samples = [p for p in self.data if p[2] == wid and len(p[1]) > 3]
            negative_samples = [n for n in self.data if n[2] != wid and len(n[1]) > 3]
            # Fallback if no samples with length > 3
            if len(positive_samples) == 0:
                positive_samples = [p for p in self.data if p[2] == wid]
            if len(negative_samples) == 0:
                negative_samples = [n for n in self.data if n[2] != wid]
            # Ensure we have samples
            if len(positive_samples) == 0:
                positive_samples = [self.data[index]]
            if len(negative_samples) == 0:
                negative_samples = [self.data[(index + 1) % len(self.data)]]
            positive_path = random.choice(positive_samples)[0]
            positive = self.load_and_verify_image(positive_path)
            if positive is None:
                positive = Image.new('RGB', self.fixed_size, (255, 255, 255))
            negative_path = random.choice(negative_samples)[0]
            negative = self.load_and_verify_image(negative_path)
            if negative is None:
                negative = Image.new('RGB', self.fixed_size, (255, 255, 255))
            # Get 5 style images (same font)
            positive_samples_for_style = [p for p in self.data if p[2] == wid]
            k = min(5, len(positive_samples_for_style))
            if k > 0:
                random_samples = random.sample(positive_samples_for_style, k=k)
                style_paths = [i[0] for i in random_samples]
                while len(style_paths) < 5:
                    style_paths.append(style_paths[0])
            else:
                style_paths = [img_path] * 5
            fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
            # Process images
            img = self._process_image(img, fheight, fwidth)
            positive = self._process_image(positive, fheight, fwidth)
            negative = self._process_image(negative, fheight, fwidth)
            # Process style images
            st_imgs = []
            for s_path in style_paths:
                s_img = self.load_and_verify_image(s_path)
                if s_img is None:
                    s_img = Image.new('RGB', self.fixed_size, (255, 255, 255))
                s_img = self._process_image(s_img, fheight, fwidth)
                if self.transforms is not None:
                    s_img_tensor = self.transforms(s_img)
                else:
                    s_img_tensor = transforms.ToTensor()(s_img)
                st_imgs.append(s_img_tensor)
            s_imgs = torch.stack(st_imgs)
            # Apply transforms to main images
            if self.transforms is not None:
                img = self.transforms(img)
                positive = self.transforms(positive)
                negative = self.transforms(negative)
            else:
                img = transforms.ToTensor()(img)
                positive = transforms.ToTensor()(positive)
                negative = transforms.ToTensor()(negative)
            # Character tokens
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
            # Return a dummy sample
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