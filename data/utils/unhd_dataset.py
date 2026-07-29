# """
# Fixed UNHD Dataset Loader for DiffusionPen
# Handles grayscale images and truncated image errors properly
# """

# import os
# import torch
# import numpy as np
# from PIL import Image, ImageOps, ImageFile
# from torch.utils.data import Dataset
# from tqdm import tqdm
# import json
# import random
# from torchvision import transforms

# # CRITICAL: Allow loading of truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True


# def image_resize_PIL(img, height=None, width=None):
#     """Resize PIL image maintaining aspect ratio"""
#     if height is None and width is None:
#         return img
    
#     orig_width, orig_height = img.size
    
#     # Handle zero dimensions
#     if orig_height == 0 or orig_width == 0:
#         return img
    
#     if height is not None and width is None:
#         ratio = height / orig_height
#         width = max(1, int(orig_width * ratio))
#     elif width is not None and height is None:
#         ratio = width / orig_width
#         height = max(1, int(orig_height * ratio))
    
#     # Ensure dimensions are at least 1x1
#     height = max(1, height)
#     width = max(1, width)
    
#     try:
#         return img.resize((width, height), Image.LANCZOS)
#     except Exception as e:
#         print(f"Warning: Error resizing image: {e}")
#         return img


# def centered_PIL(img, target_size, border_value=255):
#     """Center image on a canvas of target_size"""
#     target_h, target_w = target_size
    
#     # Convert grayscale to RGB if needed
#     if img.mode == 'L':
#         img = img.convert('RGB')
#     elif img.mode != 'RGB':
#         img = img.convert('RGB')
    
#     canvas = Image.new('RGB', (target_w, target_h), color=(border_value, border_value, border_value))
    
#     img_w, img_h = img.size
#     paste_x = max(0, (target_w - img_w) // 2)
#     paste_y = max(0, (target_h - img_h) // 2)
    
#     try:
#         canvas.paste(img, (paste_x, paste_y))
#     except Exception as e:
#         print(f"Warning: Error pasting image: {e}")
    
#     return canvas


# class UNHDDataset(Dataset):
#     def __init__(self, basefolder, subset, segmentation_level, fixed_size=(64, 256), 
#                  tokenizer=None, text_encoder=None, feat_extractor=None, transforms=None, args=None):
#         self.basefolder = basefolder
#         self.subset = subset
#         self.segmentation_level = segmentation_level
#         self.fixed_size = fixed_size
#         self.transforms = transforms
#         self.args = args
#         self.setname = 'UNHD'
        
#         # Load splits
#         self.trainset_file = './unhd_split/train_val.uttlist'
#         self.valset_file = './unhd_split/validation.uttlist'
#         self.testset_file = './unhd_split/test.uttlist'
        
#         # Load data
#         self.data = self.main_loader(subset, segmentation_level)
        
#         # Get writer info
#         self.initial_writer_ids = [d[2] for d in self.data]
#         writer_ids, _ = np.unique([d[2] for d in self.data], return_inverse=True)
#         self.writer_ids = writer_ids
#         self.wclasses = len(writer_ids)
#         print(f'Number of writers: {self.wclasses}')
        
#         # Character classes
#         res = set()
#         self.max_transcr_len = 0
#         for _, transcr, _, _ in tqdm(self.data, desc="Building character classes"):
#             res.update(list(transcr))
#             self.max_transcr_len = max(self.max_transcr_len, len(transcr))
        
#         res = sorted(list(res))
#         res.append(' ')
#         self.character_classes = res
#         print(f'Character classes: {len(res)} different characters')
#         print(f'Max transcription length: {self.max_transcr_len}')

#     def load_and_verify_image(self, img_path):
#         """Load image with proper error handling and conversion"""
#         try:
#             # Open image
#             img = Image.open(img_path)
            
#             # Verify the image can be loaded
#             img.verify()
            
#             # Reopen after verify (verify closes the file)
#             img = Image.open(img_path)
            
#             # Force load the image data
#             img.load()
            
#             # Convert grayscale to RGB
#             if img.mode == 'L':
#                 img = img.convert('RGB')
#             elif img.mode == 'RGBA':
#                 # Handle transparency by creating white background
#                 background = Image.new('RGB', img.size, (255, 255, 255))
#                 background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
#                 img = background
#             elif img.mode != 'RGB':
#                 img = img.convert('RGB')
            
#             # Check valid dimensions
#             if img.width == 0 or img.height == 0:
#                 return None
            
#             return img
            
#         except Exception as e:
#             print(f'Error loading image {img_path}: {e}')
#             return None

#     def main_loader(self, subset, segmentation_level):
#         # Load valid set
#         if subset == 'train':
#             valid_set = np.loadtxt(self.trainset_file, dtype=str)
#         elif subset == 'val':
#             valid_set = np.loadtxt(self.valset_file, dtype=str)
#         elif subset == 'test':
#             valid_set = np.loadtxt(self.testset_file, dtype=str)
#         else:
#             raise ValueError(f"Unknown subset: {subset}")
        
#         if segmentation_level != 'line':
#             raise ValueError("UNHD supports 'line' level only")
        
#         root_path = self.basefolder
#         print('root_path', root_path)
        
#         # Load writer dict
#         dict_path = './writers_dict.json'
#         with open(dict_path, 'r') as f:
#             wr_dict = json.load(f)
        
#         # Gather file info
#         gt = []
#         all_files = [f for f in os.listdir(root_path) if f.endswith('.png')]
        
#         print(f"Found {len(all_files)} PNG files")
        
#         for i, filename in enumerate(all_files):
#             if i % 1000 == 0:
#                 print(f'Processing files: [{i}/{len(all_files)} ({100. * i / len(all_files):.0f}%)]')
            
#             name = filename.replace('.png', '')
#             form_name = '_'.join(name.split('_')[:-1])
            
#             if form_name not in valid_set:
#                 continue
            
#             img_path = os.path.join(root_path, filename)
#             gt_path = os.path.join(root_path, name + '.gt.txt')
            
#             if not os.path.exists(gt_path):
#                 continue
            
#             try:
#                 with open(gt_path, 'r', encoding='utf-8') as f:
#                     transcr = f.read().strip()
#             except Exception as e:
#                 print(f"Error reading {gt_path}: {e}")
#                 continue
            
#             writer_name = name.split('_')[0]
#             writer_id = wr_dict.get(writer_name, -1)
            
#             if writer_id == -1:
#                 continue
            
#             gt.append((img_path, transcr, writer_id))
        
#         print(f"Found {len(gt)} valid samples")
        
#         # Load images with verification
#         data = []
#         failed_count = 0
        
#         for i, (img_path, transcr, writer_id) in enumerate(gt):
#             if i % 1000 == 0:
#                 print(f'Loading images: [{i}/{len(gt)} ({100. * i / len(gt):.0f}%)] - Failed: {failed_count}')
            
#             img = self.load_and_verify_image(img_path)
            
#             if img is None:
#                 failed_count += 1
#                 continue
            
#             # Store as PIL Image
#             data.append((img, transcr, writer_id, img_path))
        
#         print(f'Successfully loaded {len(data)} samples ({failed_count} failed)')
#         return data

#     def __len__(self):
#         return len(self.data)

#     def _process_image(self, img, fheight, fwidth):
#         """Process image to fixed size with proper resizing and padding"""
#         # Ensure img is PIL Image
#         if not isinstance(img, Image.Image):
#             try:
#                 img = Image.fromarray(img)
#             except:
#                 # Create white image as fallback
#                 return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
#         # Verify image is loaded
#         try:
#             img.load()
#         except:
#             return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
#         # Convert to RGB if needed
#         if img.mode == 'L':
#             img = img.convert('RGB')
#         elif img.mode == 'RGBA':
#             background = Image.new('RGB', img.size, (255, 255, 255))
#             try:
#                 background.paste(img, mask=img.split()[3])
#             except:
#                 background.paste(img)
#             img = background
#         elif img.mode != 'RGB':
#             img = img.convert('RGB')
        
#         # Check for valid dimensions
#         if img.width == 0 or img.height == 0:
#             return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
        
#         # Random resize for training
#         if self.subset == 'train':
#             nwidth = int(np.random.uniform(.75, 1.25) * img.width)
#             nheight = int((np.random.uniform(.9, 1.1) * img.height / max(img.width, 1)) * nwidth)
#         else:
#             nheight, nwidth = img.height, img.width
        
#         # Constrain size
#         nheight = max(4, min(fheight-16, nheight))
#         nwidth = max(8, min(fwidth-32, nwidth))
        
#         # Resize to target dimensions
#         img = image_resize_PIL(img, height=nheight, width=nwidth)
        
#         # Center on canvas with fixed size
#         img = centered_PIL(img, (fheight, fwidth), border_value=255)
        
#         return img

#     def __getitem__(self, index):
#         try:
#             img = self.data[index][0]
#             transcr = self.data[index][1]
#             wid = self.data[index][2]
#             img_path = self.data[index][3]
            
#             # Get positive and negative samples
#             positive_samples = [p for p in self.data if p[2] == wid and len(p[1]) > 3]
#             negative_samples = [n for n in self.data if n[2] != wid and len(n[1]) > 3]
            
#             # Fallback if no samples with length > 3
#             if len(positive_samples) == 0:
#                 positive_samples = [p for p in self.data if p[2] == wid]
#             if len(negative_samples) == 0:
#                 negative_samples = [n for n in self.data if n[2] != wid]
            
#             # Ensure we have samples
#             if len(positive_samples) == 0:
#                 positive_samples = [self.data[index]]
#             if len(negative_samples) == 0:
#                 negative_samples = [self.data[(index + 1) % len(self.data)]]
            
#             positive = random.choice(positive_samples)[0]
#             negative = random.choice(negative_samples)[0]
            
#             # Get 5 style images
#             if len(positive_samples) >= 5:
#                 random_samples = random.sample(positive_samples, k=5)
#                 style_images = [i[0] for i in random_samples]
#             else:
#                 positive_samples_ = [p for p in self.data if p[2] == wid]
#                 k = min(5, len(positive_samples_))
#                 if k > 0:
#                     random_samples_ = random.sample(positive_samples_, k=k)
#                     style_images = [i[0] for i in random_samples_]
#                     while len(style_images) < 5:
#                         style_images.append(style_images[0])
#                 else:
#                     style_images = [img] * 5
            
#             fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
            
#             # Process images
#             img = self._process_image(img, fheight, fwidth)
#             positive = self._process_image(positive, fheight, fwidth)
#             negative = self._process_image(negative, fheight, fwidth)
            
#             # Process style images
#             st_imgs = []
#             for s_im in style_images:
#                 s_img = self._process_image(s_im, fheight, fwidth)
                
#                 if self.transforms is not None:
#                     s_img_tensor = self.transforms(s_img)
#                 else:
#                     s_img_tensor = transforms.ToTensor()(s_img)
#                 st_imgs.append(s_img_tensor)
            
#             s_imgs = torch.stack(st_imgs)
            
#             # Apply transforms
#             if self.transforms is not None:
#                 img = self.transforms(img)
#                 positive = self.transforms(positive)
#                 negative = self.transforms(negative)
#             else:
#                 img = transforms.ToTensor()(img)
#                 positive = transforms.ToTensor()(positive)
#                 negative = transforms.ToTensor()(negative)
            
#             # Character tokens
#             char_tokens = [self.character_classes.index(c) if c in self.character_classes else 0 
#                           for c in transcr]
#             pad_token = len(self.character_classes) - 1
#             padding_length = 95 - len(char_tokens)
#             if padding_length > 0:
#                 char_tokens.extend([pad_token] * padding_length)
#             else:
#                 char_tokens = char_tokens[:95]
#             char_tokens = torch.tensor(char_tokens, dtype=torch.long)
            
#             return img, transcr, char_tokens, wid, positive, negative, self.character_classes, s_imgs, img_path
            
#         except Exception as e:
#             print(f"Error in __getitem__ at index {index}: {e}")
#             # Return a dummy sample
#             fheight, fwidth = self.fixed_size[0], self.fixed_size[1]
#             dummy_img = Image.new('RGB', (fwidth, fheight), (255, 255, 255))
#             if self.transforms is not None:
#                 dummy_tensor = self.transforms(dummy_img)
#             else:
#                 dummy_tensor = transforms.ToTensor()(dummy_img)
            
#             dummy_transcr = " "
#             dummy_wid = 0
#             dummy_tokens = torch.zeros(95, dtype=torch.long)
#             dummy_style = torch.stack([dummy_tensor] * 5)
            
#             return dummy_tensor, dummy_transcr, dummy_tokens, dummy_wid, dummy_tensor, dummy_tensor, [' '], dummy_style, ""

#     def collate_fn(self, batch):
#         img, transcr, char_tokens, wid, positive, negative, cla, s_imgs, img_path = zip(*batch)
        
#         images_batch = torch.stack(img)
#         char_tokens_batch = torch.stack(char_tokens)
#         images_pos = torch.stack(positive)
#         images_neg = torch.stack(negative)
#         s_imgs_batch = torch.stack(s_imgs)
        
#         return images_batch, transcr, char_tokens_batch, wid, images_pos, images_neg, cla[0], s_imgs_batch, img_path


# unhd_dataset.py
# Minor changes: Ensure RTL handling if needed, but since images are as is, no change. Already provided by user, no major changes needed.
""" Fixed UNHD Dataset Loader for DiffusionPen
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

class UNHDDataset(Dataset):
    def __init__(self, basefolder, subset, segmentation_level, fixed_size=(64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=None, args=None):
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = (64, 1024)
        self.transforms = transforms
        self.args = args
        self.setname = 'UNHD'

        # Load splits
        self.trainset_file = './unhd_split/train_val.uttlist'
        self.valset_file = './unhd_split/validation.uttlist'
        self.testset_file = './unhd_split/test.uttlist'

        # Load data
        self.data = self.main_loader(subset, segmentation_level)

        # Get writer info
        self.initial_writer_ids = [d[2] for d in self.data]
        writer_ids, _ = np.unique([d[2] for d in self.data], return_inverse=True)
        self.writer_ids = writer_ids
        self.wclasses = len(writer_ids)
        print(f'Number of writers: {self.wclasses}')

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

    def main_loader(self, subset, segmentation_level):
        # Load valid set
        if subset == 'train':
            valid_set = np.loadtxt(self.trainset_file, dtype=str)
        elif subset == 'val':
            valid_set = np.loadtxt(self.valset_file, dtype=str)
        elif subset == 'test':
            valid_set = np.loadtxt(self.testset_file, dtype=str)
        else:
            raise ValueError(f"Unknown subset: {subset}")

        if segmentation_level != 'line':
            raise ValueError("UNHD supports 'line' level only")

        root_path = self.basefolder
        print('root_path', root_path)

        # Load writer dict
        dict_path = './writers_dict.json'
        with open(dict_path, 'r') as f:
            wr_dict = json.load(f)

        # Gather file info
        gt = []
        all_files = [f for f in os.listdir(root_path) if f.endswith('.png')]
        print(f"Found {len(all_files)} PNG files")

        for i, filename in enumerate(all_files):
            if i % 1000 == 0:
                print(f'Processing files: [{i}/{len(all_files)} ({100. * i / len(all_files):.0f}%)]')
            name = filename.replace('.png', '')
            form_name = '_'.join(name.split('_')[:-1])
            if form_name not in valid_set:
                continue
            img_path = os.path.join(root_path, filename)
            gt_path = os.path.join(root_path, name + '.gt.txt')
            if not os.path.exists(gt_path):
                continue
            try:
                with open(gt_path, 'r', encoding='utf-8') as f:
                    transcr = f.read().strip()
            except Exception as e:
                print(f"Error reading {gt_path}: {e}")
                continue
            writer_name = name.split('_')[0]
            writer_id = wr_dict.get(writer_name, -1)
            if writer_id == -1:
                continue
            gt.append((img_path, transcr, writer_id))

        print(f"Found {len(gt)} valid samples")

        # Load images with verification
        data = []
        failed_count = 0
        for i, (img_path, transcr, writer_id) in enumerate(gt):
            if i % 1000 == 0:
                print(f'Loading images: [{i}/{len(gt)} ({100. * i / len(gt):.0f}%)] - Failed: {failed_count}')
            img = self.load_and_verify_image(img_path)
            if img is None:
                failed_count += 1
                continue
            # Store as PIL Image
            data.append((img, transcr, writer_id, img_path))

        print(f'Successfully loaded {len(data)} samples ({failed_count} failed)')
        return data

    def __len__(self):
        return len(self.data)

    # def _process_image(self, img, fheight, fwidth):
    #     """Process image to fixed size with proper resizing and padding"""
    #     # Ensure img is PIL Image
    #     if not isinstance(img, Image.Image):
    #         try:
    #             img = Image.fromarray(img)
    #         except:
    #             # Create white image as fallback
    #             return Image.new('RGB', (fwidth, fheight), (255, 255, 255))
    #     # Verify image is loaded
    #     try:
    #         img.load()
    #     except:
    #         return Image.new('RGB', (fwidth, fheight), (255, 255, 255))

    #     # Convert to RGB if needed
    #     if img.mode == 'L':
    #         img = img.convert('RGB')
    #     elif img.mode == 'RGBA':
    #         background = Image.new('RGB', img.size, (255, 255, 255))
    #         try:
    #             background.paste(img, mask=img.split()[3])
    #         except:
    #             background.paste(img)
    #         img = background
    #     elif img.mode != 'RGB':
    #         img = img.convert('RGB')

    #     # Check for valid dimensions
    #     if img.width == 0 or img.height == 0:
    #         return Image.new('RGB', (fwidth, fheight), (255, 255, 255))

    #     # Random resize for training
    #     if self.subset == 'train':
    #         nwidth = int(np.random.uniform(.75, 1.25) * img.width)
    #         nheight = int((np.random.uniform(.9, 1.1) * img.height / max(img.width, 1)) * nwidth)
    #     else:
    #         nheight, nwidth = img.height, img.width

    #     # Constrain size
    #     nheight = max(4, min(fheight-16, nheight))
    #     nwidth = max(8, min(fwidth-32, nwidth))

    #     # Resize to target dimensions
    #     img = image_resize_PIL(img, height=nheight, width=nwidth)

    #     # Center on canvas with fixed size
    #     img = centered_PIL(img, (fheight, fwidth), border_value=255)

    #     return img

    # def _process_image(self, img, fheight, fwidth):
    #     """
    #     Process image to fixed size with aspect-ratio preserving resize and padding.
    #     Keeps the original handwriting style (no binarization),
    #     just scales it to fit nicely on the canvas.
    #     """
    #     # Ensure img is a PIL Image
    #     if not isinstance(img, Image.Image):
    #         try:
    #             img = Image.fromarray(img)
    #         except Exception:
    #             # Fallback: white canvas
    #             return Image.new('RGB', (fwidth, fheight), (255, 255, 255))

    #     # Verify image is loaded
    #     try:
    #         img.load()
    #     except Exception:
    #         return Image.new('RGB', (fwidth, fheight), (255, 255, 255))

    #     # Convert to RGB if needed
    #     if img.mode == 'L':
    #         img = img.convert('RGB')
    #     elif img.mode == 'RGBA':
    #         background = Image.new('RGB', img.size, (255, 255, 255))
    #         try:
    #             background.paste(img, mask=img.split()[3])
    #         except Exception:
    #             background.paste(img)
    #         img = background
    #     elif img.mode != 'RGB':
    #         img = img.convert('RGB')

    #     # Check for valid dimensions
    #     if img.width == 0 or img.height == 0:
    #         return Image.new('RGB', (fwidth, fheight), (255, 255, 255))

    #     orig_w, orig_h = img.size

    #     # ---------------------------------------------------
    #     # Aspect-ratio preserving resize to almost fill canvas
    #     # ---------------------------------------------------
    #     # Usable area inside canvas (small margins top/bottom & sides)
    #     max_h = max(4, fheight - 4)   # e.g. 60 if fheight = 64
    #     max_w = max(8, fwidth  - 8)   # e.g. 248 if fwidth  = 256

    #     # Scale so BOTH dimensions fit inside usable area
    #     scale_h = max_h / float(orig_h)
    #     scale_w = max_w / float(orig_w)
    #     base_scale = min(scale_h, scale_w)

    #     # Optional tiny jitter (if you want augmentation). For now: none.
    #     if self.subset == 'train':
    #         # comment out jitter to keep geometry very stable
    #         scale = base_scale
    #     else:
    #         scale = base_scale

    #     target_h = max(4, int(orig_h * scale))

    #     # Resize with only height specified → keeps aspect ratio
    #     img = image_resize_PIL(img, height=target_h, width=None)

    #     # ---------------------------------------------------
    #     # Center on fixed canvas (no change in handwriting style)
    #     # ---------------------------------------------------
    #     img = centered_PIL(img, (fheight, fwidth), border_value=255)

    #     return img

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
        max_w = max(8, fwidth  - 8)

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
            img = self.data[index][0]
            transcr = self.data[index][1]
            wid = self.data[index][2]
            img_path = self.data[index][3]

            # Get positive and negative samples
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

            positive = random.choice(positive_samples)[0]
            negative = random.choice(negative_samples)[0]

            # Get 5 style images
            if len(positive_samples) >= 5:
                random_samples = random.sample(positive_samples, k=5)
                style_images = [i[0] for i in random_samples]
            else:
                positive_samples_ = [p for p in self.data if p[2] == wid]
                k = min(5, len(positive_samples_))
                if k > 0:
                    random_samples_ = random.sample(positive_samples_, k=k)
                    style_images = [i[0] for i in random_samples_]
                    while len(style_images) < 5:
                        style_images.append(style_images[0])
                else:
                    style_images = [img] * 5

            fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

            # Process images
            img = self._process_image(img, fheight, fwidth)
            positive = self._process_image(positive, fheight, fwidth)
            negative = self._process_image(negative, fheight, fwidth)

            # Process style images
            st_imgs = []
            for s_im in style_images:
                s_img = self._process_image(s_im, fheight, fwidth)
                if self.transforms is not None:
                    s_img_tensor = self.transforms(s_img)
                else:
                    s_img_tensor = transforms.ToTensor()(s_img)
                st_imgs.append(s_img_tensor)
            s_imgs = torch.stack(st_imgs)

            # Apply transforms
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