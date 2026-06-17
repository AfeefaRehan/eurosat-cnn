"""
EuroSAT — Step 7-9: Train + Evaluate  (EfficientNetB0 transfer learning)
Run inside your venv (after data_prep.py):   python train_efficientnet.py

Same cleaned data (data/processed/), same pipeline as the MobileNetV2 version.
Only the backbone changes: EfficientNetB0 (slightly higher accuracy, a bit heavier).

IMPORTANT: EfficientNet does its OWN normalization inside the model, so it expects
RAW [0,255] pixels (NOT [-1,1] like MobileNetV2). efficientnet.preprocess_input is a
pass-through that keeps this correct.

Outputs (in reports/):
  - training_curves_effnet.png
  - confusion_matrix_effnet.png
  - classification_report_effnet.txt
  - sample_prediction_effnet.png
Saved model: eurosat_efficientnetb0.keras
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # pass-through
from sklearn.metrics import confusion_matrix, classification_report

# ---------------- CONFIG ----------------
DATA    = Path("data/processed")
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)
IMG_SIZE = (int(os.environ.get("IMG", 128)),) * 2   # 128 = faster on CPU; 224 = designed size (best, slower)
BATCH    = int(os.environ.get("BATCH", 32))
EPOCHS   = int(os.environ.get("EPOCHS", 12))
SEED     = 42
tf.random.set_seed(SEED)

print(f"TensorFlow {tf.__version__} | EfficientNetB0 | input {IMG_SIZE} | batch {BATCH} | epochs {EPOCHS}")
print("GPU:", tf.config.list_physical_devices("GPU") or "none (running on CPU)")

# ---------------- LOAD DATA ----------------
def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="categorical", shuffle=shuffle, seed=SEED)

train_ds = load("train", True)
val_ds   = load("val",   False)
test_ds  = load("test",  False)
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ---------------- MODEL ----------------
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augment")

base = EfficientNetB0(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False                          # freeze pretrained layers

inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augment(inputs)                             # augmentation (train only); pixels stay [0,255]
x = preprocess_input(x)                         # EfficientNet: pass-through (normalization is inside base)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------- TRAIN ----------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("eurosat_efficientnetb0.keras",
                                       monitor="val_accuracy", save_best_only=True),
]
print("\n=== Training (slow part on CPU; let it run) ===")
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---------------- CURVES ----------------
h = hist.history
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(h["accuracy"], "-o", label="train"); a1.plot(h["val_accuracy"], "-o", label="val")
a1.set_title("Accuracy (EfficientNetB0)"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
a2.plot(h["loss"], "-o", label="train"); a2.plot(h["val_loss"], "-o", label="val")
a2.set_title("Loss (EfficientNetB0)"); a2.set_xlabel("epoch"); a2.legend(); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(REPORTS / "training_curves_effnet.png", dpi=130); plt.close()

# ---------------- EVALUATE ----------------
print("\n=== Evaluate on TEST set ===")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc*100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(1)
y_pred = model.predict(test_ds).argmax(1)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Greens")
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title(f"Confusion Matrix — EfficientNetB0 (test acc {test_acc*100:.1f}%)")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=7)
plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix_effnet.png", dpi=130); plt.close()

report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("\n" + report)
(REPORTS / "classification_report_effnet.txt").write_text(
    f"Model: EfficientNetB0\nTest accuracy: {test_acc*100:.2f}%\n\n{report}")

# ---------------- SAMPLE PREDICTION ----------------
for xb, yb in test_ds.take(1):
    img = xb[0].numpy().astype("uint8")
    true_lbl = class_names[int(yb[0].numpy().argmax())]
    prob = model.predict(xb[:1])[0]
    pred_lbl = class_names[int(prob.argmax())]
    plt.figure(figsize=(4, 4))
    plt.imshow(img); plt.axis("off")
    plt.title(f"True: {true_lbl}\nPred: {pred_lbl} ({prob.max()*100:.1f}%)",
              color="green" if pred_lbl == true_lbl else "red")
    plt.tight_layout(); plt.savefig(REPORTS / "sample_prediction_effnet.png", dpi=130); plt.close()
    break

print("\n*** Training complete (EfficientNetB0). ***")
print("Saved: eurosat_efficientnetb0.keras")
print("Reports: training_curves_effnet.png, confusion_matrix_effnet.png, "
      "classification_report_effnet.txt, sample_prediction_effnet.png")
