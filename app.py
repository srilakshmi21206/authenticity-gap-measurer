import os
import pandas as pd
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