"""
Builds streamlit_app/evolution_bundle.npz for the dashboard's "Auto evolve" button.
 
Run from the project root (D:\\Authenticity_Gap_Project):   python make_evolution_bundle.py
Compresses the DINOv2 embeddings to 64 dimensions (PCA, float16) so the live button runs in
under a minute on free hosting and the file stays small. Labels are stored ONLY so the dashboard
can report accuracy afterwards; the evolution itself never uses them.
"""
import os
import re
 
import numpy as np
from sklearn.decomposition import PCA
 
BASE = os.environ.get("AGM_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("AGM_OUT", os.path.join(BASE, "streamlit_app", "evolution_bundle.npz"))
SOURCES = [("No crop (v1)", "features_v1"),
           ("Gentle crop (v3)", "features"),
           ("Aggressive crop (v2)", "features_v2_aggressive")]
DIMS = 64
 
 
def lines(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
 
 
def base(p):
    return re.split(r"[\\/]", p)[-1]
 
 
f0 = os.path.join(BASE, SOURCES[0][1])
real = np.load(os.path.join(f0, "real_feats.npy")).astype(np.float32)
labels = np.load(os.path.join(f0, "real_labels.npy"))
rpaths = lines(os.path.join(f0, "real_paths.txt"))
name_to_idx = {base(p): i for i, p in enumerate(rpaths)}
classes = sorted(np.unique(labels).tolist())
class_names = [re.split(r"[\\/]", rpaths[int(np.where(labels == c)[0][0])])[-2].replace("___", " ").replace("_", " ")
               for c in classes]
 
pca = PCA(n_components=DIMS, random_state=0).fit(real)
print(f"PCA keeps {100 * pca.explained_variance_ratio_.sum():.1f}% of the variance")
pack = {"real": pca.transform(real).astype(np.float16), "real_labels": labels.astype(np.int16),
        "class_names": np.array(class_names), "n_sources": np.array(len(SOURCES)),
        "source_names": np.array([s[0] for s in SOURCES])}
 
for s, (label, folder) in enumerate(SOURCES):
    fd = os.path.join(BASE, folder)
    r2 = np.load(os.path.join(fd, "real_feats.npy"))
    if not np.allclose(r2, real, atol=1e-4):
        raise SystemExit(f"{folder} has different real features than {SOURCES[0][1]}; cannot combine.")
    sf = np.load(os.path.join(fd, "synthetic_feats.npy")).astype(np.float32)
    sl = np.load(os.path.join(fd, "synthetic_labels.npy"))
    sp = lines(os.path.join(fd, "synthetic_paths.txt"))
    src = np.array([name_to_idx.get(re.sub(r"^synth_\d+_", "", base(p)), -1) for p in sp], dtype=np.int32)
    print(f"{label}: {len(sf)} synthetic images, linked to a real source: {100 * (src >= 0).mean():.1f}%  (e.g. {sp[0]})")
    pack[f"syn{s}"] = pca.transform(sf).astype(np.float16)
    pack[f"syn{s}_src"] = src
    pack[f"syn{s}_labels"] = sl.astype(np.int16)
 
os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, **pack)
print(f"Saved {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")