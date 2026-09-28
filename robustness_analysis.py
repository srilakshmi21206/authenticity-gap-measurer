"""
robustness_analysis.py
Usage:
    python robustness_analysis.py                        # uses features/ and report/
    python robustness_analysis.py features_v1 report_v1  # before results
Outputs (in the report folder): robustness_grid.csv, robustness_excess.csv,
    robustness_heatmap.png, per_class_gap.csv, per_class_gap.png
"""
import os
import sys
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "features")
REPORT_DIR = os.path.join(ROOT, sys.argv[2] if len(sys.argv) > 2 else "report")
os.makedirs(REPORT_DIR, exist_ok=True)

real_root = os.path.join(ROOT, "real")
CLASS_NAMES = sorted(
    d for d in os.listdir(real_root) if os.path.isdir(os.path.join(real_root, d))
)


def cname(c):
    c = int(c)
    return CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)


FRACS = [0.05, 0.10, 0.15, 0.20]   # share of each class treated as edge cases
PCTS = [90, 95, 99]                # threshold percentile of normal real images
BASE_FRAC, BASE_PCT = 0.10, 95     # settings used in the main pipeline


def load(name):
    p = os.path.join(FEAT_DIR, name)
    if not os.path.isfile(p):
        sys.exit(f"Missing file: {p}")
    return np.load(p, allow_pickle=True).ravel()


labels = load("real_labels.npy")
scores = load("edge_scores.npy").astype(float)
flags = load("edge_flags.npy").astype(bool)
nn = load("nn_dist.npy").astype(float)
n = len(labels)
for nm, a in (("edge_scores", scores), ("edge_flags", flags), ("nn_dist", nn)):
    if len(a) != n:
        sys.exit(f"{nm}.npy has length {len(a)} but real_labels.npy has {n}.")

# make sure a higher score means "more of an outlier"
if scores[flags].mean() < scores[~flags].mean():
    scores = -scores

classes = np.unique(labels)


def pick_edges(frac, rounder):
    edge = np.zeros(n, dtype=bool)
    for c in classes:
        idx = np.where(labels == c)[0]
        k = max(1, int(rounder(frac * len(idx))))
        edge[idx[np.argsort(-scores[idx])[:k]]] = True
    return edge


# find the rounding rule that reproduces the saved edge flags at 10%
rounder, rname = int, "int"
for nm, f in (("int", int), ("round", round), ("ceil", math.ceil)):
    if np.array_equal(pick_edges(BASE_FRAC, f), flags):
        rounder, rname = f, nm
        break
else:
    print("Warning: could not exactly reproduce edge_flags.npy at 10%; results are approximate.")
print(f"Reading from: {FEAT_DIR}")
print(f"Edge-flag reproduction rule: {rname}")


def gap(frac, pct):
    edge = pick_edges(frac, rounder)
    unc = np.zeros(n, dtype=bool)
    per = []
    for c in classes:
        m = labels == c
        thr = np.percentile(nn[m & ~edge], pct)
        u = m & edge & (nn > thr)
        unc |= u
        per.append((cname(c), int((m & edge).sum()), int(u.sum())))
    return 100 * unc.sum() / edge.sum(), int(edge.sum()), int(unc.sum()), per, unc


base_score, base_edges, base_unc, base_per, base_mask = gap(BASE_FRAC, BASE_PCT)
print(f"Recomputed base score: {base_score:.1f}  ({base_unc} of {base_edges})")

up = os.path.join(FEAT_DIR, "uncovered.npy")
if os.path.isfile(up):
    saved = np.load(up, allow_pickle=True).ravel().astype(bool)
    if len(saved) == n:
        print("Matches saved uncovered.npy exactly:",
              bool(np.array_equal(saved, base_mask)), f"(saved count {int(saved.sum())})")

# ---------------- robustness grid ----------------
grid = pd.DataFrame(index=[f"{p}th pct" for p in PCTS],
                    columns=[f"top {int(round(f * 100))}%" for f in FRACS], dtype=float)
for p in PCTS:
    for f in FRACS:
        grid.loc[f"{p}th pct", f"top {int(round(f * 100))}%"] = round(gap(f, p)[0], 1)
grid.to_csv(os.path.join(REPORT_DIR, "robustness_grid.csv"))
print("\nAuthenticity Gap Score for different settings:")
print(grid.to_string())

# excess over what normal images would show by chance at each cutoff
excess = grid.copy()
for p in PCTS:
    excess.loc[f"{p}th pct"] = (grid.loc[f"{p}th pct"] - (100 - p)).round(1)
excess.to_csv(os.path.join(REPORT_DIR, "robustness_excess.csv"))
print("\nExcess over the chance level (score minus 100 - percentile):")
print(excess.to_string())

fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(grid.values.astype(float), cmap="Reds", vmin=0)
ax.set_xticks(range(len(grid.columns)))
ax.set_xticklabels(grid.columns)
ax.set_yticks(range(len(grid.index)))
ax.set_yticklabels(grid.index)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, f"{grid.values[i, j]:.1f}", ha="center", va="center")
ax.set_title("Gap score vs edge fraction and threshold")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "robustness_heatmap.png"), dpi=150)
plt.close(fig)

# ---------------- per-class table ----------------
pc = pd.DataFrame(base_per, columns=["class", "edge_cases", "uncovered"])
pc["gap_score"] = (100 * pc["uncovered"] / pc["edge_cases"]).round(1)
pc = pc.sort_values("gap_score", ascending=False)
pc.to_csv(os.path.join(REPORT_DIR, "per_class_gap.csv"), index=False)
print("\nPer-class gap (base settings):")
print(pc.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(pc["class"].values[::-1], pc["gap_score"].values[::-1], color="#d9534f")
ax.set_xlabel("Gap score (% of edge cases uncovered)")
ax.set_title("Where the synthetic data falls short, by class")
fig.tight_layout()
fig.savefig(os.path.join(REPORT_DIR, "per_class_gap.png"), dpi=150)
plt.close(fig)

print(f"\nChance level: normal images exceed the {BASE_PCT}th-percentile cutoff "
      f"{100 - BASE_PCT}% of the time by design.")
print("Done. Files written to", REPORT_DIR)