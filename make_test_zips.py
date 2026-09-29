import os
import zipfile

BASE = r"D:\Authenticity_Gap_Project"
OUT_DIR = os.path.join(BASE, "test_upload")
os.makedirs(OUT_DIR, exist_ok=True)


def make_zip(src_dir, out_zip, n=25):
    files = [f for f in sorted(os.listdir(src_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n]
    with zipfile.ZipFile(out_zip, "w") as zf:
        for f in files:
            zf.write(os.path.join(src_dir, f), arcname=f)
    print(f"Wrote {len(files)} images to {out_zip}")


make_zip(os.path.join(BASE, "real", "Potato___Early_blight"),
          os.path.join(OUT_DIR, "test_real.zip"))
make_zip(os.path.join(BASE, "synthetic", "Potato___Early_blight"),
          os.path.join(OUT_DIR, "test_synthetic.zip"))