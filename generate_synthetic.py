import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# Paths
REAL_DIR = r"D:\Authenticity_Gap_Project\real"          # Change if your path is different
SYNTHETIC_DIR = r"D:\Authenticity_Gap_Project\synthetic"

# Create synthetic folders
os.makedirs(SYNTHETIC_DIR, exist_ok=True)

# Strong augmentation pipeline (to make it look "synthetic")
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.4),
    A.CLAHE(p=0.3),
    A.RandomGamma(p=0.3),
])

def process_class(class_name):
    real_class_path = os.path.join(REAL_DIR, class_name)
    synth_class_path = os.path.join(SYNTHETIC_DIR, class_name)
    os.makedirs(synth_class_path, exist_ok=True)

    images = [f for f in os.listdir(real_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing: {class_name} ({len(images)} images)")

    for img_name in tqdm(images):
        img_path = os.path.join(real_class_path, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation 2 times per image (to create more synthetic samples)
        for i in range(2):
            augmented = transform(image=image)['image']
            save_name = f"synth_{i}_{img_name}"
            save_path = os.path.join(synth_class_path, save_name)
            cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

# Run for all classes
classes = os.listdir(REAL_DIR)
for cls in classes:
    if os.path.isdir(os.path.join(REAL_DIR, cls)):
        process_class(cls)

print("\nSynthetic data generation completed!")