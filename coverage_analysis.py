import os
import csv
import numpy as np
from PIL import Image

BASE = r"D:\Authenticity_Gap_Project"
FEAT = os.path.join(BASE, "features")
OUT = os.path.join(BASE, "report")
os.makedirs(OUT, exist_ok=True)

classes = sorted(
    d for d in os.listdir(os.path.join(BASE, "real"))
    if os.path.isdir(os.path.join(BASE, "real", d))
)

rf = np.load(os.path.join(FEAT, "real_feats.npy"))
rl = np.load(os.path.join(FEAT, "real_labels.npy"))
sf = np.load(os.path.join(FEAT, "synthetic_feats.npy"))
sl = np.load(os.path.join(FEAT, "synthetic_labels.npy"))
edge = np.load(os.path.join(FEAT, "edge_flags.npy"))
with open(os.path.join(FEAT, "real_paths.txt")) as f:
    paths = f.read().splitlines()

rf = rf / np.linalg.norm(rf, axis=1, keepdims=True)
sf = sf / np.linalg.norm(sf, axis=1, keepdims=True)

# distance to nearest synthetic image of the same class
nn_dist = np.zeros(len(rf), dtype=np.float32)
for ci in range(len(classes)):
    ridx = np.where(rl == ci)[0]
    sidx = np.where(sl == ci)[0]
    for s in range(0, len(ridx), 500):
        chunk = ridx[s:s + 500]
        sim = rf[chunk] @ sf[sidx].T
        nn_dist[chunk] = 1.0 - sim.max(axis=1)

# per-class threshold: 95th percentile of normal (non-edge) real images
uncovered = np.zeros(len(rf), dtype=bool)
rows = []
print(f"{'class':28s} {'edge':>5s} {'uncovered':>10s} {'gap%':>7s}")
for ci, cname in enumerate(classes):
    ridx = np.where(rl == ci)[0]
    normal = ridx[~edge[ridx]]
    thr = np.percentile(nn_dist[normal], 95)
    e_idx = ridx[edge[ridx]]
    unc = e_idx[nn_dist[e_idx] > thr]
    uncovered[unc] = True
    gap = 100 * len(unc) / max(1, len(e_idx))
    print(f"{cname:28s} {len(e_idx):5d} {len(unc):10d} {gap:6.1f}%")
    rows.append([cname, len(e_idx), len(unc), round(gap, 1), round(float(thr), 4)])

total_edge = int(edge.sum())
total_unc = int(uncovered.sum())
score = 100 * total_unc / total_edge
print(f"\nAUTHENTICITY GAP SCORE: {score:.1f} / 100")
print(f"Uncovered edge cases: {total_unc} of {total_edge}")

with open(os.path.join(OUT, "class_report.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "edge_cases", "uncovered", "gap_percent", "threshold"])
    w.writerows(rows)
    w.writerow(["OVERALL", total_edge, total_unc, round(score, 1), ""])

order = np.where(uncovered)[0]
order = order[np.argsort(-nn_dist[order])]
with open(os.path.join(OUT, "missing_edge_cases.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "class", "distance_to_nearest_synthetic"])
    for i in order:
        w.writerow([paths[i], classes[rl[i]], round(float(nn_dist[i]), 4)])

# gallery of the 24 worst-covered edge cases
top = order[:24]
if len(top):
    T, cols = 160, 6
    rws = (len(top) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rws * T), "white")
    for k, i in enumerate(top):
        im = Image.open(paths[i]).convert("RGB").resize((T, T))
        sheet.paste(im, ((k % cols) * T, (k // cols) * T))
    sheet.save(os.path.join(OUT, "worst_missing_gallery.png"))

np.save(os.path.join(FEAT, "nn_dist.npy"), nn_dist)
np.save(os.path.join(FEAT, "uncovered.npy"), uncovered)
print("Saved to", OUT)