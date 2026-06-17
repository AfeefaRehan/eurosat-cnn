"""
EuroSAT Land-Use Classifier — Live Demo (polished Gradio app for Hugging Face Spaces)
Adds: Soft theme, confidence warning for mixed scenes, per-class explanations,
optional example images (drop them in an  examples/  folder), and a footer.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

CLASS_NAMES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

CLASS_INFO = {
    "AnnualCrop": "Seasonal farmland (e.g. wheat, rice).",
    "Forest": "Dense tree cover / woodland.",
    "HerbaceousVegetation": "Grasslands, shrubs, low vegetation.",
    "Highway": "Roads and motorways.",
    "Industrial": "Factories, warehouses, industrial zones.",
    "Pasture": "Grazing land / open meadows.",
    "PermanentCrop": "Orchards, vineyards, long-term crops.",
    "Residential": "Housing and urban areas.",
    "River": "Rivers and streams.",
    "SeaLake": "Sea, lakes and large water bodies.",
}

# ---- load model ----
MODEL_PATH = "eurosat_efficientnetb0_finetuned.keras"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "eurosat_efficientnetb0.keras"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = tuple(model.input_shape[1:3])

def predict(img):
    if img is None:
        return {}, ""
    x = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.expand_dims(np.array(x).astype("float32"), 0)
    prob = model.predict(arr, verbose=0)[0]
    result = {CLASS_NAMES[i]: float(prob[i]) for i in range(len(CLASS_NAMES))}
    top = int(prob.argmax()); conf = float(prob[top]); name = CLASS_NAMES[top]
    if conf >= 0.60:
        note = f"### ✅ {name} — {conf*100:.1f}% confident\n{CLASS_INFO[name]}"
    else:
        note = (f"### ⚠️ Low confidence ({conf*100:.1f}%)\n"
                "This may be a **mixed scene** (e.g. a forest with a road through it). "
                "Try a cleaner top-down patch showing mostly one land type.")
    return result, note

# ---- optional example images: drop files into an  examples/  folder in the Space ----
examples = []
if os.path.isdir("examples"):
    examples = [os.path.join("examples", f) for f in sorted(os.listdir("examples"))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="blue")

with gr.Blocks(theme=theme, title="EuroSAT Land-Use Classifier") as demo:
    gr.Markdown(
        "# 🛰️ EuroSAT Satellite Land-Use Classifier\n"
        "Upload a **satellite / aerial (top-down)** image and the model predicts its "
        "land-use type. Trained on the EuroSAT dataset (Sentinel-2) with **EfficientNetB0 "
        "transfer learning + fine-tuning** — **96.1% test accuracy** across 10 classes.\n\n"
        "_Tip: use a clean top-down satellite patch (e.g. a Google Maps 'satellite' "
        "screenshot). Ground-level photos won't work — the model only knows overhead views._"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Upload a satellite / aerial image")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="Predicted land-use type")
            out_note = gr.Markdown()

    if examples:
        gr.Examples(examples=examples, inputs=inp, label="Try an example")

    gr.Markdown(
        "---\n"
        "**Classes:** AnnualCrop · Forest · HerbaceousVegetation · Highway · Industrial · "
        "Pasture · PermanentCrop · Residential · River · SeaLake\n\n"
        "Built by **Afeefa Rehan** — CNN image-classification project · "
        "EfficientNetB0 · TensorFlow/Keras · 96.1% accuracy."
    )

    btn.click(predict, inputs=inp, outputs=[out_label, out_note])
    inp.change(predict, inputs=inp, outputs=[out_label, out_note])

if __name__ == "__main__":
    demo.launch()