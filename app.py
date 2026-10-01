import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
 
HERE = Path(__file__).parent
REPORT = HERE / "report"

import sys
sys.path.insert(0, str(HERE / "streamlit_app"))
MAX_IMAGES_PER_SET = 100
INDIGO, LAV, TERRA, GOLD, TINT = "#2F2B6E", "#7F79B6", "#B85042", "#D9A441", "#ECEBF7"
REPO = "https://github.com/srilakshmi21206/authenticity-gap-measurer"
 
st.set_page_config(page_title="Authenticity Gap Measurer", page_icon="🌿", layout="wide")
 
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #2F2B6E 0%, #4B45A0 60%, #7F79B6 100%);
       border-radius: 18px; padding: 28px 34px; color: white; margin-bottom: 18px;}
.hero h1 {color: white; margin: 0 0 6px 0; font-size: 2.1rem;}
.hero p {color: #E3E1F5; margin: 0; font-size: 1.02rem;}
.pill {display:inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
       padding: 3px 12px; font-size: .78rem; margin: 0 8px 10px 0; color: white;}
.kpi {border-radius: 14px; padding: 16px 18px; border: 1px solid #DAD8EE; background: white;
      box-shadow: 0 2px 8px rgba(47,43,110,.07); height: 118px;}
.kpi.dark {background: #2F2B6E; border-color: #2F2B6E;}
.kpi .v {font-size: 2rem; font-weight: 700; color: #2F2B6E; line-height: 1.1;}
.kpi .l {font-size: .86rem; font-weight: 600; color: #5E5C82; margin-top: 4px;}
.kpi .s {font-size: .76rem; color: #8A88A8; margin-top: 2px;}
.kpi.dark .v, .kpi.dark .l {color: white;} .kpi.dark .s {color: #CFCBEA;}
.kpi.warn .v {color: #B85042;}
.card {border-radius: 14px; padding: 16px 20px; background: #F6F5FC; border: 1px solid #E1DFF2; margin-bottom: 10px;}
.card b {color: #2F2B6E;}
</style>
""", unsafe_allow_html=True)
 
 
class UserError(Exception):
    """Problem the user can fix (bad upload, missing package)."""
 
 
# ---------------------------------------------------------------- data
if not (REPORT / "class_report.csv").is_file() or not (REPORT / "missing_edge_case_labels.csv").is_file():
    st.error("Report files not found. Expected report/class_report.csv and "
             "report/missing_edge_case_labels.csv next to app.py.")
    st.stop()
 
 
def pretty(s):
    return re.sub(r"_+", " ", str(s)).strip()
 
 
@st.cache_data
def load_data():
    def opt(name, **kw):
        p = REPORT / name
        try:
            return pd.read_csv(p, **kw) if p.is_file() else None
        except Exception:
            return None
    return (pd.read_csv(REPORT / "class_report.csv"),
            pd.read_csv(REPORT / "missing_edge_case_labels.csv"),
            opt("before_after.csv"),
            opt("edge_case_tag_summary.csv"),
            opt("robustness_grid.csv", index_col=0),
            opt("robustness_excess.csv", index_col=0),
            opt("accuracy_runs.csv"))
 
 
cls, miss, ba, tags_df, grid, excess, acc = load_data()
overall = cls[cls["class"] == "OVERALL"].iloc[0]
per = cls[cls["class"] != "OVERALL"].copy()
per["label"] = per["class"].map(pretty)
score = float(overall["gap_percent"])
cm_path = REPORT / "confusion_matrix.png"
 
 
def resolve_image(p):
    p = str(p)
    cand = HERE / p
    if cand.is_file():
        return str(cand)
    if os.path.isfile(p):
        return p
    alt = HERE / "data" / Path(p.replace("\\", "/")).name
    return str(alt) if alt.is_file() else None
 
 
def gallery(items, ncols=6):
    ready = []
    for img, cap in items:
        if isinstance(img, str):
            img = resolve_image(img)
        if img is not None:
            ready.append((img, cap))
    if not ready:
        st.info("No images found for this selection.")
        return
    for i in range(0, len(ready), ncols):
        cols = st.columns(ncols)
        for col, (img, cap) in zip(cols, ready[i:i + ncols]):
            col.image(img, caption=cap, width="stretch")
 
 
def style_fig(fig, height=420):
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=50, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), legend_title_text="")
    fig.update_xaxes(gridcolor="#E6E4F3")
    return fig
 
 
def kpi(col, value, label, sub="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="v">{value}</div>'
                 f'<div class="l">{label}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)
 
 
def split_tags(s):
    return [t.strip() for t in re.split(r"[,;|]", str(s)) if t.strip() and t.strip().lower() != "nan"]
 
 
# ---------------------------------------------------------------- custom-data helpers
@st.cache_resource(show_spinner="Loading DINOv2 model (first run downloads ~90 MB)...")
def load_model():
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise UserError("This tab needs torch and transformers. They are not installed in the "
                        "hosted demo; run the app locally with them installed to use it.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    return processor, model
 
 
def extract_zip(file_obj, max_images):
    if file_obj is None:
        raise UserError("Please upload a .zip file for both sets.")
    tmp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(file_obj) as zf:
        names = [n for n in zf.namelist()
                 if n.lower().endswith((".jpg", ".jpeg", ".png")) and "__MACOSX" not in n]
        if not names:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise UserError("That zip has no .jpg/.png images in it.")
        names = names[:max_images]
        zf.extractall(tmp_dir, members=names)
    return [os.path.join(tmp_dir, n) for n in names], tmp_dir
 
 
def embed(paths, processor, model, batch_size=16):
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
        raise UserError("Couldn't read any valid images from that zip.")
    feats = np.concatenate(feats, axis=0)
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    return feats, good_paths
 
 
def run_custom_gap(real_file, synth_file, edge_fraction, bar):
    try:
        from sklearn.neighbors import LocalOutlierFactor
    except ImportError:
        raise UserError("This tab needs scikit-learn, which is not installed.")
    from PIL import Image
 
    bar.progress(0.02, text="Loading model...")
    processor, model = load_model()
    bar.progress(0.1, text="Reading real.zip...")
    real_paths, real_dir = extract_zip(real_file, MAX_IMAGES_PER_SET)
    bar.progress(0.2, text="Reading synthetic.zip...")
    synth_paths, synth_dir = extract_zip(synth_file, MAX_IMAGES_PER_SET)
    try:
        if len(real_paths) < 20:
            raise UserError(f"Need at least 20 real images; got {len(real_paths)}.")
        bar.progress(0.3, text=f"Embedding {len(real_paths)} real images...")
        real_feats, real_paths = embed(real_paths, processor, model)
        bar.progress(0.6, text=f"Embedding {len(synth_paths)} synthetic images...")
        synth_feats, synth_paths = embed(synth_paths, processor, model)
 
        bar.progress(0.8, text="Finding edge cases...")
        n = len(real_feats)
        k = min(20, max(2, n // 3))
        lof = LocalOutlierFactor(n_neighbors=k, metric="cosine")
        lof.fit_predict(real_feats)
        outlier_score = -lof.negative_outlier_factor_
        n_edge = max(1, int(n * edge_fraction))
        edge_mask = np.zeros(n, dtype=bool)
        edge_mask[np.argsort(outlier_score)[-n_edge:]] = True
 
        bar.progress(0.9, text="Checking synthetic coverage...")
        nn_dist = 1.0 - (real_feats @ synth_feats.T).max(axis=1)
        normal_idx = np.where(~edge_mask)[0]
        thr = np.percentile(nn_dist[normal_idx], 95) if len(normal_idx) >= 5 else np.percentile(nn_dist, 95)
        uncovered = edge_mask & (nn_dist > thr)
        gap = 100 * uncovered.sum() / edge_mask.sum()
 
        order = np.where(uncovered)[0]
        order = order[np.argsort(-nn_dist[order])]
        shown = []
        for i in order[:24]:
            im = Image.open(real_paths[i]).convert("RGB")
            im.thumbnail((256, 256))
            shown.append((im, f"distance {nn_dist[i]:.3f}"))
        summary = (f"## Your Authenticity Gap Score: {gap:.1f} / 100\n\n"
                   f"**{int(uncovered.sum())} of {int(edge_mask.sum())}** edge cases "
                   f"(rarest {int(edge_fraction * 100)}% of your {n} real images) have no good "
                   f"match among your {len(synth_paths)} synthetic images.")
        bar.progress(1.0, text="Done")
        return summary, shown
    finally:
        shutil.rmtree(real_dir, ignore_errors=True)
        shutil.rmtree(synth_dir, ignore_errors=True)
 
 
# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🌿 Authenticity Gap Measurer")
    st.caption("Problem statement #6272FBE4 · AI & IoT · आविष्KAR FET Hackathon")
    st.markdown("**How to read the score**")
    st.markdown(
        "- Find the rarest 10% of real images per class (edge cases)\n"
        "- Check whether the synthetic set has a close match for each\n"
        "- **Gap score** = % of edge cases with no close match\n"
        "- Lower is better; about 5% is chance level by design"
    )
    st.link_button("View code on GitHub", REPO, width="stretch")
    st.download_button("Download class report (CSV)", cls.to_csv(index=False),
                       file_name="class_report.csv", mime="text/csv", width="stretch")
 
# ---------------------------------------------------------------- hero + KPIs
st.markdown(
    '<div class="hero"><span class="pill">AI & IoT</span><span class="pill">Synthetic data audit</span>'
    '<span class="pill">DINOv2 + Local Outlier Factor</span>'
    '<h1>Authenticity Gap Measurer</h1>'
    '<p>Which real-world edge cases does synthetic data fail to represent, and why?</p></div>',
    unsafe_allow_html=True)
 
k1, k2, k3, k4 = st.columns(4)
kpi(k1, f"{score:.1f}<span style='font-size:1rem'> / 100</span>", "Authenticity Gap Score",
    "% of hard real images with no synthetic match", "dark")
kpi(k2, f"{int(overall['uncovered'])}<span style='font-size:1rem'> of {int(overall['edge_cases'])}</span>",
    "Edge cases not covered", f"across {len(per)} classes", "warn")
if tags_df is not None and len(tags_df):
    top = tags_df.sort_values("lift", ascending=False).iloc[0]
    kpi(k3, f"{float(top['lift']):.2f}×", "Top missing trait", pretty(top["tag"]))
else:
    kpi(k3, str(len(per)), "Classes analysed", "potato and tomato leaf diseases")
if acc is not None and "acc_syn_overall" in acc.columns:
    kpi(k4, f"{100 * acc['acc_syn_overall'].mean():.1f}%", "Accuracy, trained on synthetic only",
        f"tested on held-out real images (real-trained: {100 * acc['acc_real_overall'].mean():.1f}%)")
else:
    kpi(k4, "8,779", "Real images", "DINOv2 embeddings, 384-d")
st.write("")
 
# ---------------------------------------------------------------- tabs
tab_names = ["Overview"]
if tags_df is not None:
    tab_names.append("Why cases go missing")
tab_names.append("Missing edge cases")
if ba is not None:
    tab_names.append("Fix testing")
if acc is not None:
    tab_names.append("Accuracy")
if grid is not None:
    tab_names.append("Robustness")
tab_names += ["Auto evolve", "Try your own data", "About"]
tabs = dict(zip(tab_names, st.tabs(tab_names)))
 
# ---- Overview
with tabs["Overview"]:
    left, right = st.columns([3, 2])
    with left:
        d = per.sort_values("gap_percent")
        fig = px.bar(d, x="gap_percent", y="label", orientation="h", color="gap_percent",
                     color_continuous_scale=["#CFCBEA", INDIGO],
                     text=d["gap_percent"].map(lambda v: f"{v:.1f}%"),
                     hover_data={"edge_cases": True, "uncovered": True, "gap_percent": False, "label": False})
        fig.add_vline(x=score, line_dash="dash", line_color=TERRA,
                      annotation_text=f"Overall {score:.1f}%", annotation_position="top")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, title="Edge cases not covered, by class",
                          xaxis_title="% of edge cases uncovered", yaxis_title="")
        st.plotly_chart(style_fig(fig, 430))
        small = per[per["edge_cases"] < 30]
        if len(small):
            st.caption("Note: " + ", ".join(f"{r.label} has only {int(r.edge_cases)} edge cases"
                                              for r in small.itertuples()) + ", so its percentage is noisy.")
    with right:
        st.markdown("#### Key findings")
        st.markdown('<div class="card"><b>The gap is about framing.</b> Images with a partial or small '
                    'leaf in frame are the most over-represented among missing cases; brightness, blur '
                    'and contrast are already covered by augmentation.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>A generic metric is not enough.</b> A Fréchet-distance baseline '
                    '(0.4162 vs a 0.0042 noise floor) confirms a gap exists but cannot say which images '
                    'or why.</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><b>Fixes can be tested.</b> Random crops did not close the gap; '
                    'the per-class view shows they helped potato blights but hurt most tomato classes.</div>',
                    unsafe_allow_html=True)
 
# ---- Why cases go missing
if tags_df is not None:
    with tabs["Why cases go missing"]:
        t = tags_df.copy().sort_values("lift")
        t["label"] = t["tag"].map(pretty)
        t["status"] = np.where(t["lift"] > 1.1, "Over-represented among missing", "Not over-represented")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(t, x="lift", y="label", orientation="h", color="status",
                         color_discrete_map={"Over-represented among missing": TERRA,
                                             "Not over-represented": LAV},
                         text=t["lift"].map(lambda v: f"{v:.2f}×"))
            fig.add_vline(x=1, line_dash="dash", line_color="#555")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(title="Lift: how over-represented each trait is", xaxis_title="Lift (1.0 = no difference)",
                              yaxis_title="", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 430))
        with c2:
            if {"pct_of_missing", "pct_of_ordinary_real"}.issubset(t.columns):
                long = t.melt(id_vars="label", value_vars=["pct_of_missing", "pct_of_ordinary_real"],
                              var_name="group", value_name="pct")
                long["group"] = long["group"].map({"pct_of_missing": "Missing edge cases",
                                                   "pct_of_ordinary_real": "Ordinary real images"})
                fig = px.bar(long, x="pct", y="label", color="group", orientation="h", barmode="group",
                             color_discrete_map={"Missing edge cases": TERRA, "Ordinary real images": LAV})
                fig.update_layout(title="How common each trait is", xaxis_title="% of images", yaxis_title="",
                                  legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(style_fig(fig, 430))
        st.info("**Lift** = share of a trait among missing cases ÷ its share among ordinary real images. "
                "Tags are heuristic (brightness, sharpness, edge density), not human-verified.")
 
# ---- Missing edge cases
with tabs["Missing edge cases"]:
    miss = miss.copy()
    miss["_tags"] = miss["tags"].map(split_tags) if "tags" in miss.columns else [[] for _ in range(len(miss))]
    all_tags = sorted({t for ts in miss["_tags"] for t in ts})
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1])
    sel_cls = f1.multiselect("Class", list(per["class"]), placeholder="All classes")
    sel_tags = f2.multiselect("Trait tag", all_tags, placeholder="Any trait")
    sort_opts = ["Original order"]
    for col, lab in [("brightness", "brightness"), ("sharpness", "sharpness"), ("green_fraction", "green fraction")]:
        if col in miss.columns:
            sort_opts += [f"Lowest {lab}", f"Highest {lab}"]
    sort_by = f3.selectbox("Sort by", sort_opts)
    n_show = f4.selectbox("Show", [12, 24, 36, 60], index=1)
 
    d = miss
    if sel_cls:
        d = d[d["class"].isin(sel_cls)]
    if sel_tags:
        d = d[d["_tags"].map(lambda ts: any(t in ts for t in sel_tags))]
    if sort_by != "Original order":
        col = {"brightness": "brightness", "sharpness": "sharpness", "green fraction": "green_fraction"}[sort_by.split(" ", 1)[1]]
        d = d.sort_values(col, ascending=sort_by.startswith("Lowest"))
    st.caption(f"{len(d)} of {len(miss)} bundled images match. Click an image's corner icon to enlarge.")
    shown = d.head(int(n_show))
    gallery([(r["path"], f'{pretty(r["class"])} | {r["tags"]}') for _, r in shown.iterrows()])
 
    if len(shown):
        st.markdown("#### Inspect one image")
        names = [Path(str(p)).name for p in shown["path"]]
        pick = st.selectbox("Image", names, label_visibility="collapsed", key="inspect_pick")
        row = shown.iloc[names.index(pick)]
        i1, i2 = st.columns([1, 2])
        img = resolve_image(row["path"])
        if img:
            i1.image(img, width="stretch")
        info = {"Class": pretty(row["class"]), "Tags": ", ".join(row["_tags"]) or "none"}
        for c in ["brightness", "contrast", "sharpness", "edge_density", "green_fraction", "saturation"]:
            if c in row.index and pd.notna(row[c]):
                info[pretty(c).capitalize()] = f"{float(row[c]):.3f}"
        i2.table(pd.DataFrame({"Measure": list(info), "Value": list(info.values())}).set_index("Measure"))
 
    st.download_button("Download filtered list (CSV)",
                       d.drop(columns=["_tags", "path"], errors="ignore").to_csv(index=False),
                       file_name="missing_edge_cases_filtered.csv", mime="text/csv")
    st.caption("Only a sample of images is bundled with the hosted demo; the full analysis covers all "
               f"{int(overall['edge_cases'])} edge cases.")
 
# ---- Fix testing
if ba is not None:
    with tabs["Fix testing"]:
        st.markdown("Does adding random crops to the synthetic generator close the gap? "
                    "The per-class view shows where it helps and where it hurts.")
        plot = ba[ba["class"] != "OVERALL"].copy()
        plot["label"] = plot["class"].map(pretty)
        plot["change"] = plot["gap_percent_after"] - plot["gap_percent_before"]
        ov = ba[ba["class"] == "OVERALL"].iloc[0]
        fig = go.Figure()
        fig.add_bar(name="No crop", x=plot["label"], y=plot["gap_percent_before"], marker_color="#B2BEC3")
        fig.add_bar(name="With crop fix", x=plot["label"], y=plot["gap_percent_after"], marker_color=INDIGO)
        fig.update_layout(barmode="group", title=f"Overall: {ov['gap_percent_before']:.1f} → {ov['gap_percent_after']:.1f}",
                          yaxis_title="Gap (%)", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(style_fig(fig, 420))
        tbl = plot[["label", "gap_percent_before", "gap_percent_after", "change"]].rename(columns={
            "label": "Class", "gap_percent_before": "Before (%)", "gap_percent_after": "After (%)", "change": "Change (pts)"})
        tbl["Effect"] = np.where(tbl["Change (pts)"] < -1, "Improved", np.where(tbl["Change (pts)"] > 1, "Worse", "About flat"))
        color = {"Improved": "#D5ECD8", "Worse": "#F3D5D0", "About flat": "#F2F2F2"}
        st.dataframe(tbl.style.format({"Before (%)": "{:.1f}", "After (%)": "{:.1f}", "Change (pts)": "{:+.1f}"})
                     .map(lambda v: f"background-color: {color.get(v, '')}", subset=["Effect"]),
                     width="stretch", hide_index=True)
        st.info("The gentle crop helped both potato disease classes but hurt most tomato classes. "
                "A class with only 15 edge cases (Potato healthy) swings sharply, so treat it as noise. "
                "A single overall score hides all of this.")
 
# ---- Accuracy
if acc is not None:
    with tabs["Accuracy"]:
        st.markdown("A classifier trained **only on synthetic images** is tested on **held-out real images** "
                    "(30% of real images; synthetic training images come only from the other 70%). "
                    "Averaged over the saved runs.")
        groups = [("Overall", "overall"), ("Normal", "normal"), ("Edge, covered", "edge_covered"), ("Edge, uncovered", "edge_uncovered")]
        rows = []
        for lab, key in groups:
            s, r = f"acc_syn_{key}", f"acc_real_{key}"
            if s in acc.columns and r in acc.columns:
                rows.append((lab, 100 * acc[s].mean(), 100 * acc[r].mean()))
        if rows:
            a = pd.DataFrame(rows, columns=["Group", "Trained on synthetic", "Trained on real"])
            fig = go.Figure()
            fig.add_bar(name="Trained on synthetic", x=a["Group"], y=a["Trained on synthetic"], marker_color=INDIGO,
                        text=a["Trained on synthetic"].map("{:.1f}%".format), textposition="outside")
            fig.add_bar(name="Trained on real", x=a["Group"], y=a["Trained on real"], marker_color="#B2BEC3",
                        text=a["Trained on real"].map("{:.1f}%".format), textposition="outside")
            fig.update_layout(barmode="group", yaxis=dict(range=[70, 103], title="Accuracy (%)"),
                              title="Accuracy on held-out real images", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(style_fig(fig, 420))
            a["Cost of synthetic training (pts)"] = a["Trained on real"] - a["Trained on synthetic"]
            st.dataframe(a.style.format({"Trained on synthetic": "{:.1f}", "Trained on real": "{:.1f}",
                                         "Cost of synthetic training (pts)": "{:.1f}"}),
                         width="stretch", hide_index=True)
        if "heldout_gap_score" in acc.columns:
            st.metric("Gap score with each image's own augmentations excluded",
                      f"{acc['heldout_gap_score'].mean():.1f}",
                      help="Approximate re-implementation on a 70/30 split. Higher than the headline score, "
                           "so the headline is conservative.")
        if cm_path.is_file():
            st.markdown("#### Confusion matrices (one split)")
            st.image(str(cm_path), width="stretch")
        st.warning("The covered vs. uncovered accuracy difference is within run-to-run spread, so this is "
                   "consistent with, not proof of, the score predicting failures. The classifier is logistic "
                   "regression on frozen DINOv2 embeddings.")
 
# ---- Robustness
if grid is not None:
    with tabs["Robustness"]:
        st.markdown("Is the score an accident of one setting? Recomputed across percentile thresholds "
                    "(rows) and edge-case fractions (columns).")
        c1, c2 = st.columns(2)
        for col, df_, title in [(c1, grid, "Gap score (%)"), (c2, excess, "Excess over chance (points)")]:
            if df_ is None:
                continue
            fig = px.imshow(df_.values, x=[str(c) for c in df_.columns], y=[str(i) for i in df_.index],
                            text_auto=".1f", color_continuous_scale=["#ECEBF7", INDIGO], aspect="auto")
            fig.update_layout(title=title, coloraxis_showscale=False)
            col.plotly_chart(style_fig(fig, 300))
        st.info("About 5% of normal images exceed a 95th-percentile cutoff by design, so the excess over "
                "chance is the fairer number. The gap stays well above chance in every setting.")

# ---- Auto evolve
with tabs["Auto evolve"]:
    st.markdown(
        "This runs a **label-free** genetic algorithm: it groups real images by similarity "
        "(k-means, no class labels), finds the rarest images in each group (Local Outlier Factor), "
        "then evolves how much of each synthetic generator to use per group — under the same total "
        "image budget as the baseline — to minimize the gap score. Labels are used only afterwards, "
        "to report accuracy on held-out real images (validation, not optimisation)."
    )
    bundle_path = HERE / "streamlit_app" / "evolution_bundle.npz"
    if not bundle_path.is_file():
        st.error(f"Missing {bundle_path}. Run `python make_evolution_bundle.py` from the project "
                 "root first, then restart the app.")
    else:
        from evolution_unsupervised import (load_bundle, prepare, split_rows, pure,
                                            evolve_iter, evaluate_methods, mixture_table)

        c1, c2, c3 = st.columns(3)
        k = c1.slider("Number of groups (k)", 3, 12, 7)
        gens = c2.slider("Generations", 10, 100, 30, step=10)
        seed = c3.number_input("Seed", value=0, step=1)

        if st.button("🧬 Run Auto Evolve", type="primary"):
            data = load_bundle(str(bundle_path))
            bar = st.progress(0.0, text="Starting...")
            prep = prepare(data, k=k, seed=int(seed),
                           progress=lambda f, m: bar.progress(min(f, 0.5), text=m))
            tr, te = split_rows(prep, int(seed))

            chart_ph = st.empty()
            hist = []
            best_levels = None
            for state in evolve_iter(prep, tr, gens=gens, seed=int(seed)):
                hist.append({"Generation": state["gen"], "Best (train)": state["best"],
                            "Mean (train)": state["mean"]})
                best_levels = state["best_levels"]
                bar.progress(0.5 + 0.4 * state["gen"] / gens, text=f"Generation {state['gen']}/{gens}")
                if state["gen"] % 2 == 0 or state["done"]:
                    chart_ph.line_chart(pd.DataFrame(hist).set_index("Generation"))

            bar.progress(0.95, text="Scoring held-out data...")
            methods = {data["names"][s]: pure(prep, s) for s in range(prep["S"])}
            methods["Evolved mix"] = best_levels
            results = evaluate_methods(prep, data, methods, tr, te)
            bar.progress(1.0, text="Done")
            bar.empty()

            st.markdown("#### Held-out results (label-free evolution; accuracy is validation only)")
            st.dataframe(pd.DataFrame(results).style.format(
                {"Held-out gap score (%)": "{:.1f}", "Accuracy on held-out real (%)": "{:.1f}",
                 "Synthetic images used": "{:d}"}), width="stretch", hide_index=True)

            st.markdown("#### Evolved mixture per group")
            st.dataframe(pd.DataFrame(mixture_table(prep, data, best_levels)),
                         width="stretch", hide_index=True)        
 
# ---- Try your own data
with tabs["Try your own data"]:
    st.markdown(
        "Upload your own real and synthetic image sets as `.zip` files (flat `.jpg`/`.png`) "
        f"and get your own Authenticity Gap Score. Capped at {MAX_IMAGES_PER_SET} images per set; "
        "expect roughly 1-3 minutes on CPU."
    )
    st.info("This score measures whether the *rarest* images in your real set are worse-covered "
            "than the *typical* ones. Upload matched real and synthetic sets from the same domain.")
    u1, u2 = st.columns(2)
    real_up = u1.file_uploader("real.zip", type=["zip"])
    synth_up = u2.file_uploader("synthetic.zip", type=["zip"])
    frac = st.slider("Edge-case fraction", 0.05, 0.30, 0.10, step=0.05)
    if st.button("Compute my Authenticity Gap Score", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        try:
            st.session_state["custom_result"] = run_custom_gap(real_up, synth_up, frac, bar)
        except UserError as e:
            st.session_state.pop("custom_result", None)
            st.error(str(e))
        finally:
            bar.empty()
    if "custom_result" in st.session_state:
        summary, shown_imgs = st.session_state["custom_result"]
        st.markdown(summary)
        gallery(shown_imgs)
 
# ---- About
with tabs["About"]:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("#### Method")
        st.markdown(
            "1. **Embed** every real and synthetic image with DINOv2 (`dinov2-small`, 384-d)\n"
            "2. **Find edge cases**: Local Outlier Factor flags the rarest 10% of real images per class\n"
            "3. **Check coverage**: cosine distance to the nearest synthetic image vs. a per-class "
            "95th-percentile threshold set on normal images\n"
            "4. **Score**: % of edge cases uncovered = Authenticity Gap Score\n"
            "5. **Explain**: heuristic tags (dark, blurry, partial leaf...) say why each case is missing\n"
            "6. **Validate**: robustness grid and accuracy on held-out real images"
        )
    with a2:
        st.markdown("#### Limitations")
        st.markdown(
            "- Synthetic images are augmentations of the real ones, not an independent generator\n"
            "- A real image's nearest synthetic match can be its own augmented copy, which makes the "
            "headline score conservative\n"
            "- Tags are heuristic, not human-verified\n"
            "- The score is best used to compare classes, datasets or fixes, not as an absolute grade\n"
            "- Tiny classes (Potato healthy, 15 edge cases) are noisy"
        )
    st.link_button("Code, README and full results on GitHub", REPO)