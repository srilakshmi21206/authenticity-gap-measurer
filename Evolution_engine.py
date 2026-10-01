"""
Evolutionary engine for the Authenticity Gap Measurer.
 
Idea: the gap score diagnoses WHAT synthetic data misses. This engine uses a genetic
algorithm to DO something about it: it evolves, per class, how much of each existing
synthetic generator (no crop / gentle crop / aggressive crop) to use, under the same
size budget as the baseline synthetic set, so that the gap score is as low as possible.
 
Fitness = Authenticity Gap Score with every real image's OWN augmentations excluded
(the honest version of the score). The GA is evolved on 70% of the real images and
judged on the held-out 30%, repeated over several random splits.
 
Run from the project root (D:\\Authenticity_Gap_Project):   python evolution_engine.py
Needs the feature folders features_v1 / features / features_v2_aggressive (already
produced by extract_features.py). Writes results into report_v1/.
"""
import os
import re
import time
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
BASE = os.environ.get("AGM_BASE", os.path.dirname(os.path.abspath(__file__)))
# (label, feature folder). Check the printed synthetic paths to confirm each one.
SOURCES = [("No crop (v1)", "features_v1"),
           ("Gentle crop (v3)", "features"),
           ("Aggressive crop (v2)", "features_v2_aggressive")]
OUT = os.path.join(BASE, "report_v1")
 
L = 10            # each (class, source) gene = how many tenths of that source's class pool to use
POP, GENS, ELITE = 60, 60, 2
RUNS = 5          # independent GA runs, each with its own train/test split
TEST_FRAC = 0.30
SEED = 42
 
S = len(SOURCES)
 
 
def norm(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)
 
 
def read_lines(p):
    return [l.strip() for l in open(p, encoding="utf-8") if l.strip()]
 
 
def real_name(path):
    return re.split(r"[\\/]", path)[-1]
 
 
# ------------------------------------------------------------------ load real data
f0 = os.path.join(BASE, SOURCES[0][1])
Xr = norm(np.load(os.path.join(f0, "real_feats.npy")).astype(np.float32))
yr = np.load(os.path.join(f0, "real_labels.npy"))
edge = np.load(os.path.join(f0, "edge_flags.npy")).astype(bool)
real_paths = read_lines(os.path.join(f0, "real_paths.txt"))
name_to_idx = {real_name(p): i for i, p in enumerate(real_paths)}
classes = sorted(np.unique(yr).tolist())
C = len(classes)
N = len(Xr)
cls_names = {c: re.split(r"[\\/]", real_paths[int(np.where(yr == c)[0][0])])[-2] for c in classes}
print(f"Real images: {N}, edge cases: {int(edge.sum())}, classes: {C}")
 
 
# ------------------------------------------------------------------ precompute prefix distances
def prefix_distances(folder, seed):
    """D[l, i] = cosine distance from real image i to its nearest synthetic image of the same
    class, using the first l/L of that source's class pool and EXCLUDING i's own augmentations.
    sizes[ci][l] = number of synthetic images used for class ci at level l."""
    fdir = os.path.join(BASE, folder)
    Xs = norm(np.load(os.path.join(fdir, "synthetic_feats.npy")).astype(np.float32))
    ys = np.load(os.path.join(fdir, "synthetic_labels.npy"))
    sp = read_lines(os.path.join(fdir, "synthetic_paths.txt"))
    print(f"  {folder}: {len(Xs)} synthetic images, e.g. {sp[0]}")
    src = np.array([name_to_idx.get(re.sub(r"^synth_\d+_", "", real_name(p)), -1) for p in sp])
    if (src >= 0).mean() < 0.99:
        raise SystemExit(f"Could not link synthetic images in {folder} to real images "
                         f"(matched {100 * (src >= 0).mean():.1f}%). Send me a few lines of its paths file.")
    rng = np.random.default_rng(seed)
    D = np.full((L + 1, N), np.inf, dtype=np.float32)
    sizes = []
    for c in classes:
        rows = np.where(yr == c)[0]
        rowpos = -np.ones(N, dtype=int)
        rowpos[rows] = np.arange(len(rows))
        sidx = rng.permutation(np.where(ys == c)[0])
        bounds = np.linspace(0, len(sidx), L + 1).astype(int)
        sizes.append(bounds.copy())
        Rc = Xr[rows]
        best = np.full(len(rows), -np.inf, dtype=np.float32)
        for l in range(1, L + 1):
            chunk = sidx[bounds[l - 1]:bounds[l]]
            if len(chunk):
                sim = Rc @ Xs[chunk].T
                s_chunk = src[chunk]
                own = np.where(s_chunk >= 0, rowpos[np.maximum(s_chunk, 0)], -1)
                cols = np.where(own >= 0)[0]
                sim[own[cols], cols] = -np.inf          # leave-own-augmentation-out
                best = np.maximum(best, sim.max(axis=1))
            D[l, rows] = 1.0 - best
    return D, sizes
 
 
t0 = time.time()
print("Precomputing distances (one-off)...")
Ds, SIZES = [], []
for k, (label, folder) in enumerate(SOURCES):
    D, sizes = prefix_distances(folder, SEED + k)
    Ds.append(D)
    SIZES.append(sizes)
print(f"  done in {time.time() - t0:.0f}s")
BUDGET = sum(SIZES[0][ci][L] for ci in range(C))          # size of the baseline (v1) synthetic set
print(f"Budget: {BUDGET} synthetic images (same as the no-crop baseline)")
 
 
# ------------------------------------------------------------------ fitness
def detail(levels, rowsets):
    unc = np.zeros(C, dtype=int)
    tot = np.zeros(C, dtype=int)
    for ci, idx in enumerate(rowsets):
        nn = np.min([Ds[s][levels[ci, s], idx] for s in range(S)], axis=0)
        ed = edge[idx]
        nor = nn[~ed]
        if len(nor) == 0 or not np.isfinite(nor).all():
            return None
        thr = np.percentile(nor, 95)
        unc[ci] = int(np.sum(nn[ed] > thr))
        tot[ci] = int(ed.sum())
    return unc, tot
 
 
def gap(levels, rowsets):
    d = detail(levels, rowsets)
    if d is None:
        return 200.0
    return 100.0 * d[0].sum() / max(d[1].sum(), 1)
 
 
def size_of(levels):
    return sum(SIZES[s][ci][levels[ci, s]] for ci in range(C) for s in range(S))
 
 
def repair(levels, rng):
    """Shrink random genes until the set fits the budget; every class keeps at least one source."""
    levels = levels.copy()
    for ci in range(C):
        if levels[ci].sum() == 0:
            levels[ci, rng.integers(S)] = 1
    while size_of(levels) > BUDGET:
        cand = [(ci, s) for ci in range(C) for s in range(S)
                if levels[ci, s] > 0 and (levels[ci].sum() > 1 or levels[ci, s] > 1)]
        if not cand:
            break
        ci, s = cand[rng.integers(len(cand))]
        levels[ci, s] -= 1
    return levels
 
 
def pure(s):
    lv = np.zeros((C, S), dtype=int)
    lv[:, s] = L
    return lv
 
 
def evolve(train_rows, rng):
    pop = [repair(pure(s), rng) for s in range(S)]
    pop.append(repair(np.full((C, S), round(L / S)), rng))
    while len(pop) < POP:
        pop.append(repair(rng.integers(0, L + 1, (C, S)), rng))
    fit = np.array([gap(ind, train_rows) for ind in pop])
    history = []
    for g in range(GENS):
        order = np.argsort(fit)
        history.append((g, float(fit[order[0]]), float(fit.mean())))
        new = [pop[i].copy() for i in order[:ELITE]]
        while len(new) < POP:
            def pick():
                cand = rng.integers(0, POP, 3)
                return pop[cand[np.argmin(fit[cand])]]
            p1, p2 = pick(), pick()
            child = np.where(rng.random((C, S)) < 0.5, p1, p2) if rng.random() < 0.8 else p1.copy()
            mut = rng.random((C, S)) < 0.12
            child = np.clip(child + mut * rng.integers(-2, 3, (C, S)), 0, L)
            new.append(repair(child, rng))
        pop = new
        fit = np.array([gap(ind, train_rows) for ind in pop])
    order = np.argsort(fit)
    history.append((GENS, float(fit[order[0]]), float(fit.mean())))
    return pop[order[0]], history
 
 
def split_rows(rng):
    tr, te = [], []
    for c in classes:
        rows = rng.permutation(np.where(yr == c)[0])
        k = int(round(len(rows) * (1 - TEST_FRAC)))
        tr.append(np.sort(rows[:k]))
        te.append(np.sort(rows[k:]))
    return tr, te
 
 
# ------------------------------------------------------------------ run
labels = [s[0] for s in SOURCES]
methods = {labels[s]: pure(s) for s in range(S)}
acc = {m: {"train": [], "test": [], "unc": np.zeros(C), "tot": np.zeros(C)} for m in list(methods) + ["Evolved mix"]}
hist_rows, mix_rows = [], []
 
print(f"\nEvolving ({RUNS} runs x {GENS} generations, population {POP})...")
for run in range(RUNS):
    rng = np.random.default_rng(SEED + 100 + run)
    tr, te = split_rows(rng)
    best, history = evolve(tr, rng)
    for g, b, m in history:
        hist_rows.append({"run": run, "generation": g, "best_train_gap": b, "mean_train_gap": m})
    allm = dict(methods)
    allm["Evolved mix"] = best
    line = []
    for name, lv in allm.items():
        acc[name]["train"].append(gap(lv, tr))
        d = detail(lv, te)
        acc[name]["test"].append(100.0 * d[0].sum() / max(d[1].sum(), 1))
        acc[name]["unc"] += d[0]
        acc[name]["tot"] += d[1]
        line.append(f"{name}: {acc[name]['test'][-1]:.1f}")
    print(f"  run {run}: held-out gap  " + " | ".join(line))
    for ci, c in enumerate(classes):
        used = np.array([SIZES[s][ci][best[ci, s]] for s in range(S)], dtype=float)
        tot = used.sum()
        row = {"run": run, "class": cls_names[c], "images_used": int(tot),
               "baseline_images": int(SIZES[0][ci][L])}
        for s in range(S):
            row[f"share_{labels[s]}"] = used[s] / tot if tot else 0.0
            row[f"level_{labels[s]}"] = int(best[ci, s])
        mix_rows.append(row)
 
summary = pd.DataFrame([{
    "method": m,
    "train_gap_mean": np.mean(a["train"]),
    "heldout_gap_mean": np.mean(a["test"]),
    "heldout_gap_std": np.std(a["test"]),
    "runs": RUNS} for m, a in acc.items()])
per_class = pd.DataFrame({"class": [cls_names[c] for c in classes],
                          "heldout_edge_cases_total": acc["Evolved mix"]["tot"].astype(int)})
for m, a in acc.items():
    per_class[m] = 100.0 * a["unc"] / np.maximum(a["tot"], 1)
 
os.makedirs(OUT, exist_ok=True)
summary.to_csv(os.path.join(OUT, "evolution_summary.csv"), index=False)
pd.DataFrame(hist_rows).to_csv(os.path.join(OUT, "evolution_history.csv"), index=False)
pd.DataFrame(mix_rows).to_csv(os.path.join(OUT, "evolution_mixture.csv"), index=False)
per_class.to_csv(os.path.join(OUT, "evolution_per_class.csv"), index=False)
 
# ------------------------------------------------------------------ plots
h = pd.DataFrame(hist_rows)
g = h.groupby("generation")["best_train_gap"].agg(["mean", "min", "max"])
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(g.index, g["mean"], color="#2F2B6E", lw=2, label="Evolved mix (mean of runs)")
ax.fill_between(g.index, g["min"], g["max"], color="#7F79B6", alpha=0.3)
for m in methods:
    ax.axhline(np.mean(acc[m]["train"]), ls="--", lw=1, label=f"{m} alone")
ax.set_xlabel("Generation"); ax.set_ylabel("Gap score on training split (%)")
ax.set_title("Evolutionary search lowers the gap score"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "evolution_convergence.png"), dpi=150); plt.close(fig)
 
mix = pd.DataFrame(mix_rows).groupby("class")[[f"share_{l}" for l in labels]].mean()
fig, ax = plt.subplots(figsize=(8, 4))
bottom = np.zeros(len(mix))
for l, col in zip(labels, ["#B2BEC3", "#7F79B6", "#2F2B6E"]):
    ax.bar(mix.index, mix[f"share_{l}"] * 100, bottom=bottom, label=l, color=col)
    bottom += mix[f"share_{l}"].values * 100
ax.set_ylabel("Share of the class's synthetic set (%)"); ax.set_title("Evolved mixture per class (mean of runs)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "evolution_mixture.png"), dpi=150); plt.close(fig)
 
print("\n=== Held-out gap score (own augmentations excluded; lower is better) ===")
print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
print(f"\nSaved evolution_*.csv and evolution_*.png in {OUT}")