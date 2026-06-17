"""
EuroSAT — Step 10: FINE-TUNING (boost accuracy)
Run after train_efficientnet.py:   python fine_tune.py

Loads the already-trained model (eurosat_efficientnetb0.keras), UNFREEZES the top
layers of the EfficientNetB0 base, and trains a bit more at a very low learning rate.
This is the step the reference paper did NOT do (they only froze the base), and it
typically lifts EuroSAT accuracy from ~94% to ~97-98%.

The original 94% model is kept; the fine-tuned one saves separately.
Outputs (reports/): *_finetuned versions of curves / confusion matrix / report / sample.
Saved model: eurosat_efficientnetb0_finetuned.keras
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
from sklearn.metrics import confusion_matrix, classification_report

DATA     = Path("data/processed")
REPORTS  = Path("reports"); REPORTS.mkdir(exist_ok=True)
SRC_MODEL = "eurosat_efficientnetb0.keras"
OUT_MODEL = "eurosat_efficientnetb0_finetuned.keras"
UNFREEZE = int(os.environ.get("UNFREEZE", 40))   # how many top layers to unfreeze
EPOCHS   = int(os.environ.get("EPOCHS", 8))
LR       = float(os.environ.get("LR", 1e-5))     # very small for fine-tuning
BATCH    = int(os.environ.get("BATCH", 32))
SEED     = 42
tf.random.set_seed(SEED)

# ---------- load the trained model ----------
print(f"Loading {SRC_MODEL} ...")
model = tf.keras.models.load_model(SRC_MODEL)
IMG_SIZE = tuple(model.input_shape[1:3])          # match the model's input size
print(f"TensorFlow {tf.__version__} | input {IMG_SIZE} | unfreeze top {UNFREEZE} | lr {LR} | epochs {EPOCHS}")
print("GPU:", tf.config.list_physical_devices("GPU") or "none (running on CPU)")

# ---------- data ----------
def load(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATA / split, image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="categorical", shuffle=shuffle, seed=SEED)

train_ds = load("train", True)
val_ds   = load("val",   False)
test_ds  = load("test",  False)
class_names = train_ds.class_names
num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# ---------- find the EfficientNet base and unfreeze its top layers ----------
base = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        base = layer
        break
if base is None:                                  # fallback: any nested model that's not 'augment'
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name != "augment":
            base = layer; break
print(f"Base model: {base.name} ({len(base.layers)} layers)")

base.trainable = True
# keep everything frozen except the top UNFREEZE layers; BatchNorm stays frozen (best practice)
for layer in base.layers[:-UNFREEZE]:
    layer.trainable = False
for layer in base.layers[-UNFREEZE:]:
    layer.trainable = not isinstance(layer, layers.BatchNormalization)

trainable = sum(int(tf.size(w)) for w in model.trainable_weights)
print(f"Trainable params now: {trainable:,}")

# ---------- recompile with a SMALL learning rate and fine-tune ----------
model.compile(optimizer=tf.keras.optimizers.Adam(LR),
              loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_accuracy",
                                       save_best_only=True),
]
print("\n=== Fine-tuning (slow on CPU; let it run) ===")
hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ---------- curves ----------
h = hist.history
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot(h["accuracy"], "-o", label="train"); a1.plot(h["val_accuracy"], "-o", label="val")
a1.set_title("Accuracy (fine-tuned)"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
a2.plot(h["loss"], "-o", label="train"); a2.plot(h["val_loss"], "-o", label="val")
a2.set_title("Loss (fine-tuned)"); a2.set_xlabel("epoch"); a2.legend(); a2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(REPORTS / "training_curves_finetuned.png", dpi=130); plt.close()

# ---------- evaluate ----------
print("\n=== Evaluate on TEST set (fine-tuned) ===")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Fine-tuned TEST accuracy: {test_acc*100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds]).argmax(1)
y_pred = model.predict(test_ds).argmax(1)

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(9, 8))
ax.imshow(cm, cmap="Purples")
ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(class_names, fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("actual")
ax.set_title(f"Confusion Matrix — fine-tuned (test acc {test_acc*100:.1f}%)")
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=7)
plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix_finetuned.png", dpi=130); plt.close()

report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
print("\n" + report)
(REPORTS / "classification_report_finetuned.txt").write_text(
    f"Model: EfficientNetB0 (fine-tuned)\nTest accuracy: {test_acc*100:.2f}%\n\n{report}")

print("\n*** Fine-tuning complete. ***")
print(f"Saved: {OUT_MODEL}")
print("Compare with the frozen model's 94.21% to show the improvement.")
