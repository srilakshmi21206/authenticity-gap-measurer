import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

F = "features_v1"          # no-crop baseline
OUT = "report_v1"
SEEDS = [0, 1, 2, 3, 4]
TEST_FRAC = 0.30

Xr = np.load(f"{F}/real_feats.npy");       yr = np.load(f"{F}/real_labels.npy")
Xs = np.load(f"{F}/synthetic_feats.npy");  ys = np.load(f"{F}/synthetic_labels.npy")
edge = np.load(f"{F}/edge_flags.npy").astype(bool)
unc = np.load(f"{F}/uncovered.npy").astype(bool)
real_paths = [l.strip() for l in open(f"{F}/real_paths.txt", encoding="utf-8") if l.strip()]
syn_paths = [l.strip() for l in open(f"{F}/synthetic_paths.txt", encoding="utf-8") if l.strip()]
assert len(real_paths) == len(Xr) and len(syn_paths) == len(Xs), "path/feature length mismatch"

base = lambda p: re.split(r"[\\/]", p)[-1]
real_src = np.array([base(p) for p in real_paths])
syn_src = np.array([re.sub(r"^synth_\d+_", "", base(p)) for p in syn_paths])
match = np.isin(syn_src, real_src).mean()
print(f"Synthetic images matched to a real source: {match*100:.1f}%")
assert match > 0.99, "file-name linking failed, send me a few lines of both *_paths.txt files"

print(f"Edge cases: {edge.sum()}, uncovered edge cases: {(edge & unc).sum()}, "
      f"uncovered normal: {(~edge & unc).sum()} of {(~edge).sum()}")

group = np.where(edge & unc, "edge_uncovered", np.where(edge, "edge_covered", "normal"))
GROUPS = ["normal", "edge_covered", "edge_uncovered"]

def norm(a): return a / np.linalg.norm(a, axis=1, keepdims=True)
Xr_n, Xs_n = norm(Xr), norm(Xs)

def model(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

rows = []
for seed in SEEDS:
    tr, te = train_test_split(np.arange(len(Xr)), test_size=TEST_FRAC,
                              stratify=yr, random_state=seed)
    keep = np.isin(syn_src, real_src[tr])   # synthetic made only from training images

    m_syn = model().fit(Xs[keep], ys[keep])
    m_real = model().fit(Xr[tr], yr[tr])
    ok_syn = m_syn.predict(Xr[te]) == yr[te]
    ok_real = m_real.predict(Xr[te]) == yr[te]

    # held-out coverage: nearest synthetic excluding the image's own augmentations
    dist = np.zeros(len(te)); uncov_ho = np.zeros(len(te), bool)
    for c in np.unique(yr):
        ti = np.where(yr[te] == c)[0]
        si = np.where(keep & (ys == c))[0]
        d = 1 - (Xr_n[te][ti] @ Xs_n[si].T).max(axis=1)
        dist[ti] = d
        normal_ti = ti[group[te][ti] == "normal"]
        thr = np.percentile(dist[normal_ti], 95)
        uncov_ho[ti] = d > thr

    g_te = group[te]
    rec = {"seed": seed,
           "acc_syn_overall": ok_syn.mean(), "acc_real_overall": ok_real.mean()}
    for g in GROUPS:
        m = g_te == g
        rec[f"acc_syn_{g}"] = ok_syn[m].mean()
        rec[f"acc_real_{g}"] = ok_real[m].mean()
        rec[f"n_{g}"] = m.sum()
    e = g_te != "normal"
    rec["heldout_gap_score"] = 100 * uncov_ho[e].mean()
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/accuracy_runs.csv", index=False)

print("\n=== Trained on SYNTHETIC, tested on held-out REAL (mean over seeds) ===")
print(f"Overall accuracy: {100*df.acc_syn_overall.mean():.1f}% (+/- {100*df.acc_syn_overall.std():.1f})")
for g in GROUPS:
    print(f"  {g:15s} {100*df[f'acc_syn_{g}'].mean():5.1f}% "
          f"(+/- {100*df[f'acc_syn_{g}'].std():.1f}), avg n = {df[f'n_{g}'].mean():.0f}")

print("\n=== Baseline: trained on REAL, tested on held-out REAL ===")
print(f"Overall accuracy: {100*df.acc_real_overall.mean():.1f}%")
for g in GROUPS:
    print(f"  {g:15s} {100*df[f'acc_real_{g}'].mean():5.1f}%")

print(f"\nHeld-out gap score (own augmentations excluded): "
      f"{df.heldout_gap_score.mean():.1f} (+/- {df.heldout_gap_score.std():.1f})")
print(f"Saved {OUT}/accuracy_runs.csv")