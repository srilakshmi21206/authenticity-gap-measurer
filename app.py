import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

BASE = r"D:\Authenticity_Gap_Project"
REPORT = os.path.join(BASE, "report")

cls = pd.read_csv(os.path.join(REPORT, "class_report.csv"))
miss = pd.read_csv(os.path.join(REPORT, "missing_edge_cases.csv"))
overall = cls[cls["class"] == "OVERALL"].iloc[0]
per = cls[cls["class"] != "OVERALL"].copy()
score = float(overall["gap_percent"])


def make_chart():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(per["class"], per["gap_percent"], color="#6c5ce7")
    ax.axvline(score, ls="--", color="red", label=f"Overall {score:.1f}%")
    ax.set_xlabel("Edge cases not covered by synthetic data (%)")
    ax.set_title("Authenticity gap by class")
    ax.legend()
    fig.tight_layout()
    return fig


def show(cname, n):
    d = miss if cname == "All classes" else miss[miss["class"] == cname]
    d = d.head(int(n))
    items = [
        (r["image_path"], f'{r["class"]} | dist {r["distance_to_nearest_synthetic"]}')
        for _, r in d.iterrows()
    ]
    return items, d


with gr.Blocks(title="Authenticity Gap Measurer") as demo:
    gr.Markdown(
        f"# Authenticity Gap Score: {score:.1f} / 100\n"
        f"**{int(overall['uncovered'])} of {int(overall['edge_cases'])}** real edge cases "
        f"are not represented by the synthetic data."
    )
    gr.Plot(make_chart())
    gr.Markdown("### Missing edge cases (real images synthetic data fails to cover)")
    with gr.Row():
        dd = gr.Dropdown(["All classes"] + list(per["class"]), value="All classes", label="Class")
        sl = gr.Slider(6, 60, value=24, step=6, label="Images to show")
    gal = gr.Gallery(columns=6, height=420)
    tbl = gr.Dataframe()
    dd.change(show, [dd, sl], [gal, tbl])
    sl.change(show, [dd, sl], [gal, tbl])
    demo.load(show, [dd, sl], [gal, tbl])

demo.launch()