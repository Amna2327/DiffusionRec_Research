import PIL
from PIL import Image, ImageOps
from pathlib import Path
import os

IMG_WIDTH = 256
IMG_HEIGHT = 64

def resize_pad(image):
    (img_width, img_height) = image.size
    #resize image to height 64 keeping aspect ratio
    image = image.resize((int(img_width * 64 / img_height), 64), Image.Resampling.LANCZOS)
    (img_width, img_height) = image.size
    
    # pad image if the width is less than the max width
    if img_width > IMG_WIDTH:
        image = image.resize((IMG_WIDTH, 64), Image.Resampling.LANCZOS)
    else:
        outImg = ImageOps.pad(image, size=(IMG_WIDTH, 64), color= "white")#, centering=(0,0)) uncommment to pad right
        image = outImg
    return image


iam_path = r'./Urdu_Word_Dataset/train/images'
save_dir = r'./Urdu_Word_Dataset/train/processed_images'

image_paths = [os.path.join(root, file)
               for root, _, files in os.walk(iam_path)
               for file in files]

for image_path in image_paths:
    image_name = os.path.basename(image_path)
    print(image_name)
    
    save_path = os.path.join(save_dir, image_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB': 
            image = image.convert('RGB')
        image = resize_pad(image)
        image.save(save_path)
    except PIL.UnidentifiedImageError:
        print('Error', image_path)
        continue
folder_path=Path(save_dir)
jpg_files = list(folder_path.glob("*.jpg"))
print(f"Number of .jpg files in processed_images folder is: {len(jpg_files)}")
        