import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel

BASE = r"D:\Authenticity_Gap_Project"
REAL_DIR = os.path.join(BASE, "real")
SYN_DIR = os.path.join(BASE, "synthetic")
OUT_DIR = os.path.join(BASE, "features")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/dinov2-small"
BATCH = 32
EXT = (".jpg", ".jpeg", ".png")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ImgSet(Dataset):
    def __init__(self, root, processor):
        self.processor = processor
        self.items = []
        self.classes = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
        for ci, c in enumerate(self.classes):
            folder = os.path.join(root, c)
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(EXT):
                    self.items.append((os.path.join(folder, f), ci))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return x, label


def extract(root, name, processor, model):
    ds = ImgSet(root, processor)
    dl = DataLoader(ds, batch_size=BATCH, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(dl, desc=name):
            out = model(pixel_values=x.to(device))
            feats.append(out.last_hidden_state[:, 0].cpu().numpy())  # CLS token
            labels.append(y.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    np.save(os.path.join(OUT_DIR, f"{name}_feats.npy"), feats)
    np.save(os.path.join(OUT_DIR, f"{name}_labels.npy"), labels)
    with open(os.path.join(OUT_DIR, f"{name}_paths.txt"), "w") as fh:
        fh.write("\n".join(p for p, _ in ds.items))
    print(name, feats.shape, "classes:", ds.classes)


if __name__ == "__main__":
    print("Device:", device)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    extract(REAL_DIR, "real", processor, model)
    extract(SYN_DIR, "synthetic", processor, model)
    print("Done. Saved to", OUT_DIR)