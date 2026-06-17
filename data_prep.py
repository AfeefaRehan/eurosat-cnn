"""
EuroSAT — Data Cleaning + Preprocessing  (Step 1-6, no TensorFlow needed)
Run inside your venv:   python data_prep.py

What it does:
  1. Inventory      - list classes, count images
  2. Cleaning       - remove corrupt / duplicate / tiny images, fix non-RGB
  3. EDA            - class distribution chart + sample grid (saved to reports/)
  4. Preprocessing  - resize to 224x224, convert RGB
  5. Split          - train/val/test = 70/15/15  -> data/processed/
  6. Cleaning log   - reports/cleaning_log.csv
"""
import os, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures without a display
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- CONFIG ----------------
RAW_DIR  = Path("data/raw/EuroSAT_RGB")     # <- your dataset folder
PROC_DIR = Path("data/processed")           # cleaned + resized output
REPORTS  = Path("reports")
IMG_SIZE = (224, 224)                        # MobileNetV2 input size
MIN_SIDE = 32                                # flag images smaller than this
SPLIT    = (0.70, 0.15, 0.15)                # train / val / test
SEED     = 42
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

random.seed(SEED); np.random.seed(SEED)
REPORTS.mkdir(parents=True, exist_ok=True)

if not RAW_DIR.exists():
    raise SystemExit(f"[X] Dataset not found at {RAW_DIR.resolve()}\n"
                     f"    Make sure the 10 class folders are inside it.")

def average_hash(img, n=8):
    """Tiny perceptual hash -> images that look the same get the same string."""
    g = img.convert("L").resize((n, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.float64)
    bits = (a > a.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)

# ---------------- 1. INVENTORY ----------------
print("\n=== 1. INVENTORY ===")
classes = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
print("Classes:", classes)

rows = []
for cls in classes:
    for p in (RAW_DIR / cls).iterdir():
        if p.suffix.lower() in VALID_EXT:
            rows.append({"path": str(p), "filename": p.name, "class": cls})
df = pd.DataFrame(rows)
print("Total images found:", len(df))
print(df["class"].value_counts().to_string())

# ---------------- 2. CLEANING (scan) ----------------
print("\n=== 2. CLEANING (scanning images) ===")
status, modes, widths, heights, md5s, ahashes = [], [], [], [], [], []
for i, p in enumerate(df["path"]):
    try:
        with Image.open(p) as im:
            im.verify()                      # detects corrupt files
        with Image.open(p) as im:
            mode, (w, h) = im.mode, im.size
            ah = average_hash(im)
        with open(p, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
        status.append("ok"); modes.append(mode); widths.append(w)
        heights.append(h);   md5s.append(md5); ahashes.append(ah)
    except Exception:
        status.append("corrupt"); modes.append(None); widths.append(0)
        heights.append(0); md5s.append(None); ahashes.append(None)
    if (i + 1) % 3000 == 0:
        print(f"  scanned {i+1}/{len(df)}")

df["status"] = status; df["mode"] = modes
df["width"] = widths; df["height"] = heights
df["md5"] = md5s; df["ahash"] = ahashes

# flag issues
df["issue"] = ""
df.loc[df.status == "corrupt", "issue"] = "corrupt"
ok_idx = df.index[df.status == "ok"]

tiny = (df.loc[ok_idx, "width"] < MIN_SIDE) | (df.loc[ok_idx, "height"] < MIN_SIDE)
df.loc[ok_idx[tiny.values], "issue"] = "too_small"

df["non_rgb"] = (df.status == "ok") & (df["mode"] != "RGB")   # will be fixed on resize

clean_idx = df.index[(df.status == "ok") & (df.issue == "")]
exact_dup = df.loc[clean_idx].duplicated(subset="md5", keep="first")
df.loc[clean_idx[exact_dup.values], "issue"] = "duplicate_exact"

clean_idx = df.index[(df.status == "ok") & (df.issue == "")]
near_dup = df.loc[clean_idx].duplicated(subset="ahash", keep="first")
df.loc[clean_idx[near_dup.values], "issue"] = "duplicate_near"

print("\nIssue summary:")
print(df["issue"].replace("", "good").value_counts().to_string())
print("Non-RGB images (auto-fixed on resize):", int(df["non_rgb"].sum()))

df.to_csv(REPORTS / "cleaning_log.csv", index=False)
print(f"-> cleaning log saved: {REPORTS/'cleaning_log.csv'}")

good = df[(df.status == "ok") & (df.issue == "")].copy()
print(f"Clean images kept: {len(good)} / {len(df)}")

# ---------------- 3. EDA ----------------
print("\n=== 3. EDA (charts) ===")
counts = good["class"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(11, 5))
plt.bar(counts.index, counts.values, color="#1C7293")
plt.title("Class distribution (after cleaning)")
plt.ylabel("images"); plt.xticks(rotation=40, ha="right")
plt.tight_layout(); plt.savefig(REPORTS / "class_distribution.png", dpi=130)
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax, cls in zip(axes.flat, classes):
    sample = good[good["class"] == cls]["path"].iloc[0]
    ax.imshow(Image.open(sample)); ax.set_title(cls, fontsize=9)
    ax.axis("off")
plt.tight_layout(); plt.savefig(REPORTS / "sample_grid.png", dpi=130)
plt.close()
print("-> saved: class_distribution.png, sample_grid.png")
print("Image size stats (px):  width", df.loc[ok_idx, 'width'].mode().tolist(),
      " height", df.loc[ok_idx, 'height'].mode().tolist())

# ---------------- 4 & 5. PREPROCESS + SPLIT ----------------
print("\n=== 4-5. PREPROCESS (resize 224) + SPLIT 70/15/15 ===")
for sub in ("train", "val", "test"):
    for cls in classes:
        (PROC_DIR / sub / cls).mkdir(parents=True, exist_ok=True)

split_counts = {"train": 0, "val": 0, "test": 0}
done = 0
for cls in classes:
    paths = good[good["class"] == cls]["path"].tolist()
    random.shuffle(paths)
    n = len(paths)
    n_tr = int(n * SPLIT[0]); n_va = int(n * SPLIT[1])
    buckets = {"train": paths[:n_tr],
               "val":   paths[n_tr:n_tr + n_va],
               "test":  paths[n_tr + n_va:]}
    for sub, plist in buckets.items():
        for p in plist:
            try:
                img = Image.open(p).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
                out = PROC_DIR / sub / cls / (Path(p).stem + ".jpg")
                img.save(out, "JPEG", quality=92)
                split_counts[sub] += 1
            except Exception:
                pass
            done += 1
            if done % 3000 == 0:
                print(f"  processed {done}/{len(good)}")

print("\nDone. Images per split:", split_counts)
print(f"-> cleaned + resized data saved in: {PROC_DIR.resolve()}")
print("\nNOTE: pixels normalized at TRAINING time using MobileNetV2 preprocess_input "
      "([-1,1]); saved images stay as standard RGB so they remain viewable.")
print("\n*** Data cleaning + preprocessing complete. ***")
