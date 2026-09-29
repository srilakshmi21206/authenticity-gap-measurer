import os
import zipfile

BASE = r"D:\Authenticity_Gap_Project"
OUT_DIR = os.path.join(BASE, "test_upload")


def make_zip(files, src_dir, out_zip):
    with zipfile.ZipFile(out_zip, "w") as zf:
        for f in files:
            zf.write(os.path.join(src_dir, f), arcname=f)
    print(f"Wrote {len(files)} images to {out_zip}")


real_dir = os.path.join(BASE, "real", "Potato___Early_blight")
synth_dir = os.path.join(BASE, "synthetic", "Potato___Early_blight")

real_files = sorted(f for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".png")))[:60]
# only keep synthetic copies for the FIRST 40 of those 60 real images —
# the last 20 real images genuinely have no synthetic match
covered_stems = {os.path.splitext(f)[0] for f in real_files[:40]}
synth_files = [f for f in os.listdir(synth_dir)
               if f.lower().endswith((".jpg", ".png"))
               and any(f.endswith(stem + ".jpg") or stem in f for stem in covered_stems)][:80]

make_zip(real_files, real_dir, os.path.join(OUT_DIR, "test_real_partial.zip"))
make_zip(synth_files, synth_dir, os.path.join(OUT_DIR, "test_synthetic_partial.zip"))