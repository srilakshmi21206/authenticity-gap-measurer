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
    sys.exit("synthetic/ is not empty. Rename it to synthetic_v2 first.")
os.makedirs(SYNTHETIC_DIR, exist_ok=True)


def build_transform(h, w):
    try:
        crop = A.RandomResizedCrop(size=(h, w), scale=(0.25, 0.95), ratio=(0.7, 1.4), p=0.8)
    except (TypeError, ValueError):
        crop = A.RandomResizedCrop(height=h, width=w, scale=(0.25, 0.95), ratio=(0.7, 1.4), p=0.8)

    steps = [
        # --- Strong partial / small leaf simulation ---
        crop,
        A.CoarseDropout(
            max_holes=8, max_height=h//5, max_width=w//5,
            min_holes=1, min_height=h//12, min_width=w//12,
            fill_value=0, p=0.5
        ),

        # --- Geometric deformations ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.25, rotate_limit=45, p=0.6),
        A.ElasticTransform(alpha=60, sigma=60 * 0.05, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.3),

        # --- Color & lighting ---
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
        A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, val_shift_limit=25, p=0.7),
        A.CLAHE(p=0.3),
        A.RandomGamma(p=0.3),

        # --- Noise & blur ---
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.35),
        A.MotionBlur(blur_limit=7, p=0.2),
    ]

    try:
        return A.Compose(steps, seed=42)
    except TypeError:
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

        for i in range(2):  # still generating 2 synthetic per real image
            augmented = tf(image=image)['image']
            save_path = os.path.join(synth_class_path, f"synth_{i}_{img_name}")
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))


for cls in sorted(os.listdir(REAL_DIR)):
    if os.path.isdir(os.path.join(REAL_DIR, cls)):
        process_class(cls)

print("\nSynthetic data generation v3 completed!")