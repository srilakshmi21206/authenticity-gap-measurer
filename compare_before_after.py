import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
before = pd.read_csv(os.path.join(ROOT, "report_v1", "class_report.csv"))
after = pd.read_csv(os.path.join(ROOT, "report", "class_report.csv"))

m = before[["class", "edge_cases", "uncovered", "gap_percent"]].merge(
    after[["class", "edge_cases", "uncovered", "gap_percent"]],
    on="class", suffixes=("_before", "_after"),
)
m["change"] = (m["gap_percent_after"] - m["gap_percent_before"]).round(1)
print(m.to_string(index=False))
m.to_csv(os.path.join(ROOT, "report", "before_after.csv"), index=False)

plot = m[m["class"] != "OVERALL"]
overall = m[m["class"] == "OVERALL"].iloc[0]
x = range(len(plot))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar([i - 0.2 for i in x], plot["gap_percent_before"], 0.4, label="v1 (no crop)", color="#b2bec3")
ax.bar([i + 0.2 for i in x], plot["gap_percent_after"], 0.4, label="v2 (with crop)", color="#6c5ce7")
ax.set_xticks(list(x))
ax.set_xticklabels(plot["class"], rotation=30, ha="right")
ax.set_ylabel("Edge cases not covered (%)")
ax.set_title(f"Gap score: {overall['gap_percent_before']:.1f} to {overall['gap_percent_after']:.1f}")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "report", "before_after.png"), dpi=150)
print("Saved report/before_after.csv and report/before_after.png")