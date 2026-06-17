"""
EuroSAT — Test the model on YOUR OWN images
Run:  python predict.py

1. Make a folder called  my_test_images  in your project (the script creates it if missing).
2. Put some SATELLITE / AERIAL (top-down) images in it — e.g. screenshots from
   Google Maps "satellite" view of a forest, river, city, farmland, sea, etc.
   (cropped roughly square). JPG / PNG both work.
3. Run this script. It prints the top-3 predicted land types + confidence for each
   image, and saves a visual grid to reports/my_predictions.png.

NOTE: the model only knows these 10 satellite land-types from a top-down view.
A normal ground-level photo (or a cat!) will still be forced into one of the 10
classes and give meaningless results — use overhead/satellite-style images.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

# class order MUST match training (alphabetical folder order)
CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

# prefer the fine-tuned model if it exists, else the base one
MODEL_PATH = "eurosat_efficientnetb0_finetuned.keras"
if not Path(MODEL_PATH).exists():
    MODEL_PATH = "eurosat_efficientnetb0.keras"
if not Path(MODEL_PATH).exists():
    raise SystemExit("[X] No trained model found. Run train_efficientnet.py first.")

print(f"Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = tuple(model.input_shape[1:3])          # match model input (e.g. 128x128)

Path("reports").mkdir(exist_ok=True)
IMG_DIR = Path(os.environ.get("DIR", "my_test_images"))
IMG_DIR.mkdir(exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
files = sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in exts])
if not files:
    print(f"\n[!] No images found in '{IMG_DIR}/'.")
    print("    Put some satellite/aerial images there, then run again.")
    raise SystemExit

print(f"Found {len(files)} image(s) in {IMG_DIR}/\n")
results = []
for p in files:
    img = Image.open(p).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.expand_dims(np.array(img).astype("float32"), 0)   # [0,255]; model normalizes inside
    prob = model.predict(arr, verbose=0)[0]
    top3 = prob.argsort()[-3:][::-1]
    print(f"📷 {p.name}")
    for rank, i in enumerate(top3, 1):
        bar = "█" * int(prob[i] * 20)
        print(f"    {rank}. {CLASS_NAMES[i]:22s} {prob[i]*100:5.1f}%  {bar}")
    print()
    results.append((img, CLASS_NAMES[top3[0]], float(prob[top3[0]])))

# save a visual grid
n = len(results)
cols = min(4, n); rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
axes = np.array(axes).reshape(-1)
for ax in axes:
    ax.axis("off")
for ax, (img, lbl, conf) in zip(axes, results):
    ax.imshow(img)
    ax.set_title(f"{lbl}\n{conf*100:.1f}%", fontsize=10,
                 color="green" if conf > 0.6 else "orange")
plt.tight_layout()
plt.savefig("reports/my_predictions.png", dpi=130)
print("✓ Saved visual grid: reports/my_predictions.png")
