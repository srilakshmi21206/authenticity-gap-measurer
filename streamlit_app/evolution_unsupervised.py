"""
Unsupervised evolutionary engine (used by the dashboard's "Auto evolve" button).
 
No class labels are used to cluster, find edge cases, or evolve. Labels are only used at the very
end, to REPORT accuracy on held-out real images (validation, not optimisation).
 
Pipeline
  1. k-means groups the real images (unsupervised stand-in for classes)
  2. Local Outlier Factor inside each cluster flags the rarest real images (edge cases)
  3. For every synthetic source (e.g. no crop / gentle crop / aggressive crop) precompute how well
     the first l/10 of its pool covers every real image, EXCLUDING the image's own augmentations
  4. A genetic algorithm evolves, per cluster, how much of each source to use under the same
     size budget as the baseline. Fitness = gap score (edge cases left uncovered) + a weighted
     overall-coverage term. Both are label-free.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
L = 10   # each (cluster, source) gene = how many tenths of that source's cluster pool to use
 
 
def _norm(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
 
 
def load_bundle(path):
    z = np.load(path, allow_pickle=False)
    n = int(z["n_sources"])
    return {
        "real": z["real"].astype(np.float32),
        "real_labels": z["real_labels"].astype(int),
        "class_names": [str(x) for x in z["class_names"]],
        "names": [str(x) for x in z["source_names"]],
        "syn": [z[f"syn{s}"].astype(np.float32) for s in range(n)],
        "syn_src": [z[f"syn{s}_src"].astype(int) for s in range(n)],
        "syn_lab": [z[f"syn{s}_labels"].astype(int) for s in range(n)],
    }
 
 
def prepare(data, k=7, seed=0, edge_fraction=0.10, progress=None):
    say = progress or (lambda f, m: None)
    Xr = _norm(data["real"])
    N = len(Xr)
    say(0.05, "Grouping real images without labels (k-means)...")
    cl = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Xr)
 
    say(0.2, "Finding edge cases inside each group (Local Outlier Factor)...")
    edge = np.zeros(N, dtype=bool)
    for c in range(k):
        rows = np.where(cl == c)[0]
        if len(rows) < 5:
            continue
        lof = LocalOutlierFactor(n_neighbors=min(20, len(rows) - 1), metric="cosine")
        lof.fit_predict(Xr[rows])
        score = -lof.negative_outlier_factor_
        n_e = max(1, int(round(len(rows) * edge_fraction)))
        edge[rows[np.argsort(score)[-n_e:]]] = True
 
    S = len(data["syn"])
    rng = np.random.default_rng(seed)
    D, sizes, pools = [], [], []
    for s in range(S):
        say(0.3 + 0.3 * s / S, f"Measuring coverage by '{data['names'][s]}'...")
        Xs = _norm(data["syn"][s])
        src = data["syn_src"][s]
        syn_cl = np.where(src >= 0, cl[np.maximum(src, 0)], -1)
        Ds = np.full((L + 1, N), np.inf, dtype=np.float32)
        sz, pl = [], []
        for c in range(k):
            rows = np.where(cl == c)[0]
            rowpos = -np.ones(N, dtype=int)
            rowpos[rows] = np.arange(len(rows))
            sidx = rng.permutation(np.where(syn_cl == c)[0])
            b = np.linspace(0, len(sidx), L + 1).astype(int)
            sz.append(b)
            pl.append(sidx)
            if len(rows) == 0:
                continue
            Rc = Xr[rows]
            best = np.full(len(rows), -np.inf, dtype=np.float32)
            for l in range(1, L + 1):
                ch = sidx[b[l - 1]:b[l]]
                if len(ch):
                    sim = Rc @ Xs[ch].T
                    s_ch = src[ch]
                    own = np.where(s_ch >= 0, rowpos[np.maximum(s_ch, 0)], -1)
                    cols = np.where(own >= 0)[0]
                    sim[own[cols], cols] = -np.inf          # leave-own-augmentation-out
                    best = np.maximum(best, sim.max(axis=1))
                Ds[l, rows] = 1.0 - best
        D.append(Ds)
        sizes.append(sz)
        pools.append(pl)
    budget = sum(sizes[0][c][L] for c in range(k))
    return {"k": k, "S": S, "N": N, "cl": cl, "edge": edge, "D": D, "sizes": sizes,
            "pools": pools, "budget": int(budget)}
 
 
def split_rows(prep, seed, test_frac=0.30):
    rng = np.random.default_rng(seed + 1)
    tr, te = [], []
    for c in range(prep["k"]):
        rows = rng.permutation(np.where(prep["cl"] == c)[0])
        kk = int(round(len(rows) * (1 - test_frac)))
        tr.append(np.sort(rows[:kk]))
        te.append(np.sort(rows[kk:]))
    return tr, te
 
 
def pure(prep, s):
    lv = np.zeros((prep["k"], prep["S"]), dtype=int)
    lv[:, s] = L
    return lv
 
 
def detail(prep, levels, rowsets):
    k, S = prep["k"], prep["S"]
    unc = np.zeros(k, dtype=int)
    tot = np.zeros(k, dtype=int)
    nn_sum, nn_cnt = 0.0, 0
    for c, idx in enumerate(rowsets):
        if len(idx) == 0:
            continue
        nn = np.min([prep["D"][s][levels[c, s], idx] for s in range(S)], axis=0)
        ed = prep["edge"][idx]
        nor = nn[~ed]
        if len(nor) == 0 or not np.isfinite(nn).all():
            return None
        thr = np.percentile(nor, 95)
        unc[c] = int(np.sum(nn[ed] > thr))
        tot[c] = int(ed.sum())
        nn_sum += float(nn.sum())
        nn_cnt += len(nn)
    return unc, tot, nn_sum, nn_cnt
 
 
def size_of(prep, levels):
    return sum(prep["sizes"][s][c][levels[c, s]] for c in range(prep["k"]) for s in range(prep["S"]))
 
 
def repair(prep, levels, rng):
    k, S = prep["k"], prep["S"]
    levels = levels.copy()
    for c in range(k):
        if levels[c].sum() == 0:
            levels[c, rng.integers(S)] = 1
    while size_of(prep, levels) > prep["budget"]:
        cand = [(c, s) for c in range(k) for s in range(S)
                if levels[c, s] > 0 and (levels[c].sum() > 1 or levels[c, s] > 1)]
        if not cand:
            break
        c, s = cand[rng.integers(len(cand))]
        levels[c, s] -= 1
    return levels
 
 
def evolve_iter(prep, train_rows, pop_size=40, gens=30, w_cov=25.0, seed=0, elite=2):
    """Generator: yields one dict per generation; the last one has done=True."""
    rng = np.random.default_rng(seed + 7)
    k, S = prep["k"], prep["S"]
    base = detail(prep, pure(prep, 0), train_rows)
    base_mean = (base[2] / max(base[3], 1)) if base else 1.0
 
    def fit(ind):
        d = detail(prep, ind, train_rows)
        if d is None:
            return 1e3
        gap = 100.0 * d[0].sum() / max(d[1].sum(), 1)
        return gap + w_cov * (d[2] / max(d[3], 1)) / max(base_mean, 1e-9)
 
    pop = [repair(prep, pure(prep, s), rng) for s in range(S)]
    pop.append(repair(prep, np.full((k, S), round(L / S)), rng))
    while len(pop) < pop_size:
        pop.append(repair(prep, rng.integers(0, L + 1, (k, S)), rng))
    f = np.array([fit(i) for i in pop])
    for g in range(gens + 1):
        order = np.argsort(f)
        yield {"gen": g, "best": float(f[order[0]]), "mean": float(f.mean()),
               "best_levels": pop[order[0]].copy(), "done": g == gens}
        if g == gens:
            break
        new = [pop[i].copy() for i in order[:elite]]
        while len(new) < pop_size:
            def pick():
                c3 = rng.integers(0, len(pop), 3)
                return pop[c3[np.argmin(f[c3])]]
            p1, p2 = pick(), pick()
            child = np.where(rng.random((k, S)) < 0.5, p1, p2) if rng.random() < 0.8 else p1.copy()
            mut = rng.random((k, S)) < 0.12
            child = np.clip(child + mut * rng.integers(-2, 3, (k, S)), 0, L)
            new.append(repair(prep, child, rng))
        pop = new
        f = np.array([fit(i) for i in pop])
 
 
def evaluate_methods(prep, data, methods, train_rows, test_rows):
    """Held-out unsupervised gap score and (label-based, validation only) accuracy."""
    k, S = prep["k"], prep["S"]
    train_mask = np.zeros(prep["N"], dtype=bool)
    train_mask[np.concatenate(train_rows)] = True
    test_idx = np.concatenate(test_rows)
    Xte = data["real"][test_idx]
    yte = data["real_labels"][test_idx]
    rows = []
    for name, lv in methods.items():
        d = detail(prep, lv, test_rows)
        gap = 100.0 * d[0].sum() / max(d[1].sum(), 1) if d else float("nan")
        Xs, ys = [], []
        for s in range(S):
            src = data["syn_src"][s]
            for c in range(k):
                n = prep["sizes"][s][c][lv[c, s]]
                idx = prep["pools"][s][c][:n]
                idx = idx[(src[idx] >= 0) & train_mask[np.maximum(src[idx], 0)]]
                if len(idx):
                    Xs.append(data["syn"][s][idx])
                    ys.append(data["syn_lab"][s][idx])
        acc = float("nan")
        if Xs:
            Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
            if len(np.unique(ytr)) > 1:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
                clf.fit(Xtr, ytr)
                acc = 100.0 * float((clf.predict(Xte) == yte).mean())
        rows.append({"Method": name, "Held-out gap score (%)": gap, "Accuracy on held-out real (%)": acc,
                     "Synthetic images used": int(size_of(prep, lv))})
    return rows
 
 
def mixture_table(prep, data, levels):
    k, S = prep["k"], prep["S"]
    y = data["real_labels"]
    out = []
    for c in range(k):
        rows = np.where(prep["cl"] == c)[0]
        used = np.array([prep["sizes"][s][c][levels[c, s]] for s in range(S)], dtype=float)
        tot = used.sum()
        maj = int(np.bincount(y[rows]).argmax()) if len(rows) else -1
        name = data["class_names"][maj] if 0 <= maj < len(data["class_names"]) else "?"
        rec = {"cluster": f"Cluster {c + 1} (mostly {name})", "images": len(rows)}
        for s in range(S):
            rec[data["names"][s]] = 100.0 * used[s] / tot if tot else 0.0
        out.append(rec)
    return out
 