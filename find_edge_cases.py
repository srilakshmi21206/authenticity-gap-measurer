import os
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

BASE = r"D:\Authenticity_Gap_Project"
FEAT = os.path.join(BASE, "features")
REAL_DIR = os.path.join(BASE, "real")

EDGE_FRACTION = 0.10   # top 10% rarest images per class
K = 20                 # neighbours for density estimate

feats = np.load(os.path.join(FEAT, "real_feats.npy"))
labels = np.load(os.path.join(FEAT, "real_labels.npy"))
classes = sorted(
    d for d in os.listdir(REAL_DIR) if os.path.isdir(os.path.join(REAL_DIR, d))
)

# cosine geometry: L2-normalise embeddings
feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

edge_flags = np.zeros(len(feats), dtype=bool)
edge_scores = np.zeros(len(feats), dtype=np.float32)

for ci, cname in enumerate(classes):
    idx = np.where(labels == ci)[0]
    lof = LocalOutlierFactor(n_neighbors=K, metric="cosine")
    lof.fit_predict(feats[idx])
    score = -lof.negative_outlier_factor_      # higher = rarer
    edge_scores[idx] = score
    n_edge = max(1, int(len(idx) * EDGE_FRACTION))
    top = idx[np.argsort(score)[-n_edge:]]
    edge_flags[top] = True
    print(f"{cname:28s} images: {len(idx):5d}  edge cases: {n_edge}")

np.save(os.path.join(FEAT, "edge_flags.npy"), edge_flags)
np.save(os.path.join(FEAT, "edge_scores.npy"), edge_scores)
print("Total edge cases:", int(edge_flags.sum()), "of", len(feats))
print("Saved edge_flags.npy and edge_scores.npy")