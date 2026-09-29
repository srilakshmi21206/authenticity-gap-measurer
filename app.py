import os
import zipfile
import tempfile
import shutil

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

BASE = r"D:\Authenticity_Gap_Project"
REPORT = os.path.join(BASE, "report")

cls = pd.read_csv(os.path.join(REPORT, "class_report.csv"))
miss = pd.read_csv(os.path.join(REPORT, "missing_edge_case_labels.csv"))
overall = cls[cls["class"] == "OVERALL"].iloc[0]
per = cls[cls["class"] != "OVERALL"].copy()
score = float(overall["gap_percent"])

ba_path = os.path.join(REPORT, "before_after.csv")
ba = pd.read_csv(ba_path) if os.path.isfile(ba_path) else None


def make_chart():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(per["class"], per["gap_percent"], color="#6c5ce7")
    ax.axvline(score, ls="--", color="red", label=f"Overall {score:.1f}%")
    ax.set_xlabel("Edge cases not covered by synthetic data (%)")
    ax.set_title("Authenticity gap by class")
    ax.legend()
    fig.tight_layout()
    return fig


def make_before_after_chart():
    if ba is None:
        return None
    plot = ba[ba["class"] != "OVERALL"]
    ov = ba[ba["class"] == "OVERALL"].iloc[0]
    x = range(len(plot))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], plot["gap_percent_before"], 0.4, label="No crop", color="#b2bec3")
    ax.bar([i + 0.2 for i in x], plot["gap_percent_after"], 0.4, label="With crop fix", color="#6c5ce7")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot["class"], rotation=30, ha="right")
    ax.set_ylabel("Gap (%)")
    ax.set_title(f"Overall: {ov['gap_percent_before']:.1f} -> {ov['gap_percent_after']:.1f}")
    ax.legend()
    fig.tight_layout()
    return fig


def show(cname, n):
    d = miss if cname == "All classes" else miss[miss["class"] == cname]
    d = d.head(int(n))
    items = [(r["path"], f'{r["class"]} | {r["tags"]}') for _, r in d.iterrows()]
    table_cols = ["path", "class", "tags", "brightness", "sharpness", "green_fraction"]
    return items, d[table_cols]


# ---------------- "Try your own data" tab ----------------
MAX_IMAGES_PER_SET = 100
_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        _model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return _processor, _model


def _extract_zip(file_obj, max_images):
    if file_obj is None:
        raise gr.Error("Please upload a .zip file.")
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(file_obj.name) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png"))
                 and "__MACOSX" not in n]
        if not names:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise gr.Error("That zip has no .jpg/.png images in it.")
        names = names[:max_images]
        zf.extractall(tmp_dir, members=names)
    return [os.path.join(tmp_dir, n) for n in names], tmp_dir


def _embed(paths, processor, model, batch_size=16):
    import torch
    from PIL import Image
    device = next(model.parameters()).device
    feats, good_paths, buf_imgs, buf_paths = [], [], [], []

    def flush():
        if not buf_imgs:
            return
        x = processor(images=buf_imgs, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            out = model(pixel_values=x)
        feats.append(out.last_hidden_state[:, 0].cpu().numpy())
        good_paths.extend(buf_paths)
        buf_imgs.clear()
        buf_paths.clear()

    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        buf_imgs.append(img)
        buf_paths.append(p)
        if len(buf_imgs) >= batch_size:
            flush()
    flush()
    if not feats:
        raise gr.Error("Couldn't read any valid images from that zip.")
    feats = np.concatenate(feats, axis=0)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return feats, good_paths


def run_custom_gap(real_zip, synth_zip, edge_fraction, progress=gr.Progress()):
    from sklearn.neighbors import LocalOutlierFactor

    progress(0.02, desc="Loading model (first run downloads ~90MB)...")
    processor, model = _load_model()

    progress(0.1, desc="Reading real.zip...")
    real_paths, real_dir = _extract_zip(real_zip, MAX_IMAGES_PER_SET)
    progress(0.2, desc="Reading synthetic.zip...")
    synth_paths, synth_dir = _extract_zip(synth_zip, MAX_IMAGES_PER_SET)

    try:
        if len(real_paths) < 20:
            raise gr.Error(f"Need at least 20 real images; got {len(real_paths)}.")

        progress(0.3, desc=f"Embedding {len(real_paths)} real images...")
        real_feats, real_paths = _embed(real_paths, processor, model)
        progress(0.6, desc=f"Embedding {len(synth_paths)} synthetic images...")
        synth_feats, synth_paths = _embed(synth_paths, processor, model)

        progress(0.8, desc="Finding edge cases...")
        n = len(real_feats)
        k = min(20, max(2, n // 3))
        lof = LocalOutlierFactor(n_neighbors=k, metric="cosine")
        lof.fit_predict(real_feats)
        outlier_score = -lof.negative_outlier_factor_
        n_edge = max(1, int(n * edge_fraction))
        edge_idx = np.argsort(outlier_score)[-n_edge:]
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[edge_idx] = True

        progress(0.9, desc="Checking synthetic coverage...")
        sim = real_feats @ synth_feats.T
        nn_dist = 1.0 - sim.max(axis=1)
        normal_idx = np.where(~edge_mask)[0]
        thr = np.percentile(nn_dist[normal_idx], 95) if len(normal_idx) >= 5 else np.percentile(nn_dist, 95)
        uncovered = edge_mask & (nn_dist > thr)

        gap = 100 * uncovered.sum() / edge_mask.sum()
        debug_info = ""

        order = np.where(uncovered)[0]
        order = order[np.argsort(-nn_dist[order])]
        gallery = [(real_paths[i], f"distance {nn_dist[i]:.3f}") for i in order[:24]]

        summary = (
            f"## Your Authenticity Gap Score: {gap:.1f} / 100\n\n"
            f"**{int(uncovered.sum())} of {int(edge_mask.sum())}** edge cases "
            f"(rarest {int(edge_fraction * 100)}% of your {n} real images) have no good "
            f"match among your {len(synth_paths)} synthetic images."
            + debug_info
        )
        progress(1.0, desc="Done")
        return summary, gallery
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(synth_dir, ignore_errors=True)


with gr.Blocks(title="Authenticity Gap Measurer") as demo:
    gr.Markdown(
        f"# Authenticity Gap Score: {score:.1f} / 100\n"
        f"**{int(overall['uncovered'])} of {int(overall['edge_cases'])}** real edge cases "
        f"are not represented by the synthetic data."
    )
    with gr.Tab("Gap by class"):
        gr.Plot(make_chart())
    if ba is not None:
        with gr.Tab("Before vs after crop fix"):
            gr.Plot(make_before_after_chart())
            gr.Markdown(
                "Cropping helped large classes but hurt small ones "
                "(e.g. Potato healthy, only 15 edge cases) — a single "
                "overall score hides this; the per-class view reveals it."
            )
    with gr.Tab("Try your own data"):
        gr.Markdown(
            "Upload your own real and synthetic image sets as `.zip` files "
            "(flat `.jpg`/`.png`, no folder structure needed) and get your "
            f"own Authenticity Gap Score. Capped at {MAX_IMAGES_PER_SET} "
            "images per set to keep this a live, on-the-spot demo — "
            "expect roughly 1-3 minutes on CPU."
        )
        gr.Markdown(
            "**Note:** this score measures whether the *rarest* images in your "
            "real set are worse-covered than the *typical* ones — not whether "
            "your two uploads are from completely unrelated datasets. Upload "
            "matched real/synthetic sets from the same domain for a meaningful "
            "result."
        )
        with gr.Row():
            real_up = gr.File(label="real.zip", file_types=[".zip"])
            synth_up = gr.File(label="synthetic.zip", file_types=[".zip"])
        frac_slider = gr.Slider(0.05, 0.3, value=0.10, step=0.05, label="Edge-case fraction")
        run_btn = gr.Button("Compute my Authenticity Gap Score", variant="primary")
        out_md = gr.Markdown()
        out_gallery = gr.Gallery(columns=6, height=350)
        run_btn.click(run_custom_gap, [real_up, synth_up, frac_slider], [out_md, out_gallery])

    gr.Markdown("### Missing edge cases (real images synthetic data fails to cover, with why)")
    with gr.Row():
        dd = gr.Dropdown(["All classes"] + list(per["class"]), value="All classes", label="Class")
        sl = gr.Slider(6, 60, value=24, step=6, label="Images to show")
    gal = gr.Gallery(columns=6, height=420)
    tbl = gr.Dataframe()
    dd.change(show, [dd, sl], [gal, tbl])
    sl.change(show, [dd, sl], [gal, tbl])
    demo.load(show, [dd, sl], [gal, tbl])

demo.launch()