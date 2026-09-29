import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

F, OUT = "features_v1", "report_v1"   # no-crop baseline
SEED = 0                               # same split as accuracy_check.py

Xr = np.load(f"{F}/real_feats.npy");      yr = np.load(f"{F}/real_labels.npy")
Xs = np.load(f"{F}/synthetic_feats.npy"); ys = np.load(f"{F}/synthetic_labels.npy")
real_paths = [l.strip() for l in open(f"{F}/real_paths.txt", encoding="utf-8") if l.strip()]
syn_paths  = [l.strip() for l in open(f"{F}/synthetic_paths.txt", encoding="utf-8") if l.strip()]

part = lambda p, k: re.split(r"[\\/]", p)[k]
real_src = np.array([part(p, -1) for p in real_paths])
syn_src  = np.array([re.sub(r"^synth_\d+_", "", part(p, -1)) for p in syn_paths])
classes = sorted(np.unique(yr).tolist())
names = {c: part(real_paths[int(np.where(yr == c)[0][0])], -2) for c in classes}
short = [names[c].replace("___", " ").replace("_", " ") for c in classes]

tr, te = train_test_split(np.arange(len(Xr)), test_size=0.30, stratify=yr, random_state=SEED)
keep = np.isin(syn_src, real_src[tr])          # synthetic made only from training images
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
m_syn = mk().fit(Xs[keep], ys[keep])           # trained on synthetic
m_real = mk().fit(Xr[tr], yr[tr])              # baseline: trained on real
y_te = yr[te]

results = {}
for title, m in [("Trained on SYNTHETIC", m_syn), ("Trained on REAL (baseline)", m_real)]:
    pred = m.predict(Xr[te])
    results[title] = (confusion_matrix(y_te, pred, labels=classes), accuracy_score(y_te, pred))

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (title, (cm, acc)) in zip(axes, results.items()):
    rown = cm / cm.sum(1, keepdims=True)
    ax.imshow(rown, cmap="Greens", vmin=0, vmax=1)
    ax.set_title(f"{title}\naccuracy {100*acc:.1f}% (held-out real images)")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=55, ha="right", fontsize=9); ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if rown[i, j] > 0.6 else "black")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)

cm, acc = results["Trained on SYNTHETIC"]
np.savetxt(f"{OUT}/confusion_matrix_synthetic.csv", cm, fmt="%d", delimiter=",", header=",".join(short), comments="")

print(f"Test images: {len(te)}")
print(f"Synthetic-trained accuracy: {100*acc:.1f}%   Real-trained accuracy: {100*results['Trained on REAL (baseline)'][1]:.1f}%")
print("\nPer-class recall (synthetic-trained):")
for i, c in enumerate(classes):
    print(f"  {short[i]:28s} {100*cm[i,i]/cm[i].sum():5.1f}%  ({cm[i,i]}/{cm[i].sum()})")
off = [(cm[i, j], short[i], short[j]) for i in range(len(classes)) for j in range(len(classes)) if i != j]
print("\nTop confusions (true -> predicted):")
for n, a, b in sorted(off, reverse=True)[:5]:
    print(f"  {n:3d}  {a} -> {b}")
print(f"\nSaved {OUT}/confusion_matrix.png and {OUT}/confusion_matrix_synthetic.csv")