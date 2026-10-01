"""Builds the small data bundle for the hosted Streamlit demo.
Run from D:\\Authenticity_Gap_Project:  python streamlit_app\\prepare_deploy.py
Output goes into streamlit_app\\report and streamlit_app\\data."""
import os
import shutil
import pandas as pd
 
BASE = r"D:\Authenticity_Gap_Project"
APP = os.path.join(BASE, "streamlit_app")
BASELINE = os.path.join(BASE, "report_v1")   # no-crop baseline: score 33.0
SCRATCH = os.path.join(BASE, "report")        # holds before_after.csv from the latest comparison
PER_CLASS = 22                                # images bundled per class
 
os.makedirs(os.path.join(APP, "data"), exist_ok=True)
os.makedirs(os.path.join(APP, "report"), exist_ok=True)
 
shutil.copy(os.path.join(BASELINE, "class_report.csv"), os.path.join(APP, "report", "class_report.csv"))
print("Copied class_report.csv")
 
for folder in (SCRATCH, BASELINE):
    src = os.path.join(folder, "before_after.csv")
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(APP, "report", "before_after.csv"))
        print("Copied before_after.csv from", folder)
        break
else:
    print("No before_after.csv found (the before/after tab will be hidden)")
 
for extra in ["edge_case_tag_summary.csv", "robustness_grid.csv", "robustness_excess.csv",
              "accuracy_runs.csv", "confusion_matrix.png"]:
    src = os.path.join(BASELINE, extra)
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(APP, "report", extra))
        print("Copied", extra)
    else:
        print("Skipped (not found):", extra)
 
df = pd.read_csv(os.path.join(BASELINE, "missing_edge_case_labels.csv"))
df = df.groupby("class", group_keys=False).head(PER_CLASS).reset_index(drop=True)
copied = 0
keep = []
for _, row in df.iterrows():
    src = row["path"]
    if not os.path.isfile(src):
        continue
    name = os.path.basename(src)
    dst = os.path.join(APP, "data", name)
    if not os.path.isfile(dst):
        shutil.copy(src, dst)
        copied += 1
    row["path"] = "data/" + name
    keep.append(row)
out = pd.DataFrame(keep)
out.to_csv(os.path.join(APP, "report", "missing_edge_case_labels.csv"), index=False)
print(f"Bundled {copied} images, wrote {len(out)} rows")
print(out["class"].value_counts().to_string())