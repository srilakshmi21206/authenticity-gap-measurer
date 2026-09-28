"""
label_and_baseline.py

Two jobs, both aimed at the problem statement:

1. LABEL the missing (uncovered) real edge cases  -> answers "specifically which"
   Each missing image gets tags such as dark, blurry, cluttered, partial_or_small_leaf.
   Tag rates are compared against a random sample of ordinary real images.

2. BASELINE generic real-vs-synthetic metrics      -> answers "rather than generic comparisons"
   Frechet distance in DINOv2 space, centroid distance and mean nearest-neighbour distance,
   shown next to the edge-case gap score.

Run from the project folder:   python label_and_baseline.py
Outputs go to report/.
"""
import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(ROOT, "real")
SYN_DIR = os.path.join(ROOT, "synthetic")
FEAT_DIR = os.path.join(ROOT, "features")
REPORT_DIR = os.path.join(ROOT, "report")
GAP_SCORE = 33.0          # your Authenticity Gap Score from coverage_analysis.py
REF_SAMPLE = 1500         # ordinary real images used as the tag reference
SEED = 0
EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ----------------------------------------------------------------------

os.makedirs(REPORT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)


# ------------------------- helpers ------------------------------------
def list_images(root):
    """Sorted (class, path) list: classes sorted, files sorted inside each class."""
    items = []
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith(EXTS):
                items.append((cls, os.path.join(cdir, f)))
    return items


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def image_stats(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = 256.0 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 100, 200)
    green = ((hsv[..., 0] >= 25) & (hsv[..., 0] <= 95) &
             (hsv[..., 1] > 40) & (hsv[..., 2] > 40)).mean()
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "edge_density": float((edges > 0).mean()),
        "green_fraction": float(green),
        "saturation": float(hsv[..., 1].mean()),
    }


def find_missing_csv():
    csvs = glob.glob(os.path.join(REPORT_DIR, "*.csv"))
    pref = [c for c in csvs if any(k in os.path.basename(c).lower()
                                   for k in ("missing", "uncovered"))]
    if pref:
        return pref[0]
    print("Could not find a 'missing/uncovered' CSV in report/. CSVs found:")
    for c in csvs:
        print("  ", os.path.basename(c))
    sys.exit("Set the file name manually in find_missing_csv().")


def resolve_paths(df):
    cols = {c.lower(): c for c in df.columns}
    path_col = next((cols[c] for c in cols if any(k in c for k in ("path", "file", "image", "name"))), None)
    cls_col = next((cols[c] for c in cols if any(k in c for k in ("class", "label"))), None)
    if path_col is None:
        sys.exit(f"No path/file column found. Columns are: {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        p = str(r[path_col])
        cands = [p, os.path.join(ROOT, p), os.path.join(REAL_DIR, p)]
        if cls_col is not None:
            cands.append(os.path.join(REAL_DIR, str(r[cls_col]), os.path.basename(p)))
        out.append(next((c for c in cands if os.path.isfile(c)), None))
    return out


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    eps = 1e-6 * np.eye(a.shape[1])
    s1 = np.cov(a, rowvar=False) + eps
    s2 = np.cov(b, rowvar=False) + eps
    covmean = linalg.sqrtm(s1.dot(s2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


# ======================= PART 1: LABEL MISSING EDGE CASES ================
print("\n=== PART 1: labelling missing edge cases ===")
miss_csv = find_missing_csv()
print("Using", os.path.basename(miss_csv))
miss = pd.read_csv(miss_csv)
miss["resolved_path"] = resolve_paths(miss)
n_bad = miss["resolved_path"].isna().sum()
if n_bad:
    print(f"Warning: {n_bad} rows could not be matched to image files and will be skipped.")
miss = miss.dropna(subset=["resolved_path"]).reset_index(drop=True)
print("Missing edge cases found:", len(miss))

real_items = list_images(REAL_DIR)
ref_idx = rng.choice(len(real_items), size=min(REF_SAMPLE, len(real_items)), replace=False)
ref_rows = [image_stats(real_items[i][1]) for i in tqdm(ref_idx, desc="Reference stats")]
ref = pd.DataFrame([r for r in ref_rows if r])
p10, p90 = ref.quantile(0.10), ref.quantile(0.90)

RULES = {
    "dark": lambda s: s["brightness"] < p10["brightness"],
    "overexposed": lambda s: s["brightness"] > p90["brightness"],
    "low_contrast": lambda s: s["contrast"] < p10["contrast"],
    "blurry": lambda s: s["sharpness"] < p10["sharpness"],
    "cluttered_background": lambda s: s["edge_density"] > p90["edge_density"],
    "partial_or_small_leaf": lambda s: s["green_fraction"] < p10["green_fraction"],
    "washed_out_colour": lambda s: s["saturation"] < p10["saturation"],
}

rows = []
for _, r in tqdm(miss.iterrows(), total=len(miss), desc="Labelling missing"):
    st = image_stats(r["resolved_path"])
    if st is None:
        continue
    tags = [t for t, f in RULES.items() if f(st)]
    rows.append({"path": r["resolved_path"],
                 "class": os.path.basename(os.path.dirname(r["resolved_path"])),
                 **st, "tags": ", ".join(tags) if tags else "no_visible_trait"})
lab = pd.DataFrame(rows)
lab.to_csv(os.path.join(REPORT_DIR, "missing_edge_case_labels.csv"), index=False)

summary = []
for t, f in RULES.items():
    pct_ref = 100 * ref.apply(f, axis=1).mean()
    pct_miss = 100 * lab["tags"].str.contains(t).mean()
    summary.append({"tag": t, "pct_of_missing": round(pct_miss, 1),
                    "pct_of_ordinary_real": round(pct_ref, 1),
                    "lift": round(pct_miss / pct_ref, 2) if pct_ref else np.nan})
summary.append({"tag": "no_visible_trait",
                "pct_of_missing": round(100 * (lab["tags"] == "no_visible_trait").mean(), 1),
                "pct_of_ordinary_real": np.nan, "lift": np.nan})
summ = pd.DataFrame(summary).sort_values("pct_of_missing", ascending=False)
summ.to_csv(os.path.join(REPORT_DIR, "edge_case_tag_summary.csv"), index=False)
print("\nWhat the synthetic data misses (lift > 1 = over-represented among missing):")
print(summ.to_string(index=False))

plot_df = summ[summ["tag"] != "no_visible_trait"]
x = np.arange(len(plot_df))
plt.figure(figsize=(9, 4.5))
plt.bar(x - 0.2, plot_df["pct_of_missing"], 0.4, label="Missing edge cases")
plt.bar(x + 0.2, plot_df["pct_of_ordinary_real"], 0.4, label="Ordinary real images")
plt.xticks(x, plot_df["tag"], rotation=30, ha="right")
plt.ylabel("% of images with trait")
plt.title("What kind of real images does the synthetic data miss?")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "edge_case_tags.png"), dpi=150)
plt.close()

# ======================= PART 2: GENERIC BASELINE ========================
print("\n=== PART 2: generic real-vs-synthetic baseline ===")
npys = glob.glob(os.path.join(FEAT_DIR, "*.npy"))
print("Feature files:", [os.path.basename(p) for p in npys])
real_f = next((p for p in npys if "real" in os.path.basename(p).lower()), None)
syn_f = next((p for p in npys if "syn" in os.path.basename(p).lower()), None)
if not (real_f and syn_f):
    sys.exit("Could not find real*/syn* .npy files in features/. Send me the file names and I'll adapt this.")

R, S = l2n(np.load(real_f)), l2n(np.load(syn_f))
syn_items = list_images(SYN_DIR)
aligned = (len(R) == len(real_items)) and (len(S) == len(syn_items))
print(f"Real features {R.shape}, synthetic features {S.shape}, order matches folders: {aligned}")

fd_all = frechet(R, S)
perm = rng.permutation(len(R))
half = len(R) // 2
fd_floor = frechet(R[perm[:half]], R[perm[half:]])
rc, sc = l2n(R.mean(0, keepdims=True))[0], l2n(S.mean(0, keepdims=True))[0]
centroid_cos = 1 - float(rc @ sc)

nn = NearestNeighbors(n_neighbors=1).fit(S)
d_all, _ = nn.kneighbors(R)
d_all = d_all[:, 0]
result = {
    "frechet_real_vs_synthetic": fd_all,
    "frechet_real_half_vs_half (noise floor)": fd_floor,
    "centroid_cosine_distance": centroid_cos,
    "mean_nn_distance_all_real": float(d_all.mean()),
}

if aligned:
    idx = {norm(p): i for i, (_, p) in enumerate(real_items)}
    mi = [idx[norm(p)] for p in lab["path"] if norm(p) in idx]
    if mi:
        result["mean_nn_distance_missing_edge_cases"] = float(d_all[mi].mean())
        result["missing_vs_all_ratio"] = result["mean_nn_distance_missing_edge_cases"] / result["mean_nn_distance_all_real"]
else:
    print("Skipping the missing-vs-all comparison because feature order could not be verified.")

result["edge_case_gap_score"] = GAP_SCORE
pd.Series(result).to_csv(os.path.join(REPORT_DIR, "baseline_vs_gap.csv"), header=["value"])
print()
for k, v in result.items():
    print(f"{k:45s} {v:.4f}")

if "mean_nn_distance_missing_edge_cases" in result:
    plt.figure(figsize=(5, 4))
    vals = [result["mean_nn_distance_all_real"], result["mean_nn_distance_missing_edge_cases"]]
    plt.bar(["All real images\n(generic view)", "Missing edge cases\n(our view)"], vals,
            color=["#6c8ebf", "#d9534f"])
    plt.ylabel("Distance to nearest synthetic image")
    plt.title("Averages hide the gap")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "generic_vs_edge_case.png"), dpi=150)
    plt.close()

print("\nDone. Files written to report/: missing_edge_case_labels.csv, edge_case_tag_summary.csv,"
      " edge_case_tags.png, baseline_vs_gap.csv, generic_vs_edge_case.png")