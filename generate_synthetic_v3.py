import os
import sys
import random

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

random.seed(42)
np.random.seed(42)

REAL_DIR = r"D:\Authenticity_Gap_Project\real"
SYNTHETIC_DIR = r"D:\Authenticity_Gap_Project\synthetic"

if os.path.isdir(SYNTHETIC_DIR) and any(os.scandir(SYNTHETIC_DIR)):
    sys.exit("synthetic/ is not empty. Rename it first.")
os.makedirs(SYNTHETIC_DIR, exist_ok=True)


def build_transform(h, w):
    try:
        crop = A.RandomResizedCrop(size=(h, w), scale=(0.6, 1.0), ratio=(0.85, 1.18), p=0.7)
    except (TypeError, ValueError):
        crop = A.RandomResizedCrop(height=h, width=w, scale=(0.6, 1.0), ratio=(0.85, 1.18), p=0.7)
    steps = [
        crop,  # gentler than v2: keeps 60-100% of the image, not 30-100%
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.4),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),
    ]
    try:
        return A.Compose(steps, seed=42)
    except (TypeError, ValueError):
        return A.Compose(steps)


transforms = {}


def process_class(class_name):
    real_class_path = os.path.join(REAL_DIR, class_name)
    synth_class_path = os.path.join(SYNTHETIC_DIR, class_name)
    os.makedirs(synth_class_path, exist_ok=True)
    images = [f for f in sorted(os.listdir(real_class_path))
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing: {class_name} ({len(images)} images)")
    for img_name in tqdm(images):
        image = cv2.imread(os.path.join(real_class_path, img_name))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if (h, w) not in transforms:
            transforms[(h, w)] = build_transform(h, w)
        tf = transforms[(h, w)]
        for i in range(2):
            augmented = tf(image=image)['image']
            save_path = os.path.join(synth_class_path, f"synth_{i}_{img_name}")
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


for cls in sorted(os.listdir(REAL_DIR)):
    if os.path.isdir(os.path.join(REAL_DIR, cls)):
        process_class(cls)

print("\nSynthetic data generation v3 completed!")