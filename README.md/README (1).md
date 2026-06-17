# 🛰️ EuroSAT Land-Use Classification — CNN (Transfer Learning)

Satellite land-use / land-cover classification on the EuroSAT dataset using a
Convolutional Neural Network with transfer learning (EfficientNetB0).
Given a satellite image patch, the model predicts which of 10 land types it shows.

**Final result: 96.1% test accuracy** (4,022 unseen images) · macro-F1 **0.96**
(94.2% with frozen transfer learning → **96.1% after fine-tuning**)

### 🔗 Live Demo

**Try it yourself:** https://huggingface.co/spaces/Afeefarehan/eurosat-land-classifier

Upload any satellite / aerial (top-down) image and the model predicts its land-use
type with confidence scores — deployed with Gradio on Hugging Face Spaces.

---

## 1. Problem & Use Case

Land-cover classification is a core task in remote sensing / earth observation.
Automatically labelling satellite imagery powers real applications such as:

- **Agriculture monitoring** — tracking croplands and farmland
- **Urban planning** — measuring how residential/industrial areas expand
- **Environmental monitoring** — forests, rivers, lakes, deforestation
- **Disaster response** — e.g. flood mapping (rivers / lakes / affected farmland)
- **Map updating** — keeping maps current from fresh imagery

The data (ESA Sentinel-2) is openly available, so models like this are scalable.

---

## 2. Dataset

**EuroSAT (RGB)** — Sentinel-2 satellite image patches.

| Property | Value |
|---|---|
| Total images | 27,000 |
| Classes | 10 |
| Original size | 64 × 64 px, RGB |
| Source | ESA Sentinel-2 |

Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial,
Pasture, PermanentCrop, Residential, River, SeaLake.

Mild class imbalance (2,000–3,000 images per class).

---

## 3. Data Cleaning & Preprocessing  (`data_prep.py`)

Real-world ML rests on clean data, so the pipeline first cleans, then prepares the data.

**Cleaning**

- Corrupt / unreadable image detection (removed)
- Format & colour-mode check → all converted to RGB
- Exact-duplicate detection (MD5 hash)
- Near-duplicate detection (perceptual average-hash)
- Too-small / low-quality image flagging
- Full audit saved to `reports/cleaning_log.csv`

**Cleaning summary**

| Metric | Count |
|---|---|
| Total images | 27,000 |
| Near-duplicates removed | 223 |
| Corrupt removed | 0 |
| Too-small removed | 0 |
| Clean images kept | 26,777 |

(EuroSAT is a curated benchmark, so it was already fairly clean — 0.8% removed.)

**Preprocessing**

- Resized all images to 128 px for the CNN
- Normalization handled by the model (EfficientNet built-in)
- Train / Val / Test split = 70 / 15 / 15
- Light augmentation on the training set only (flip, rotation, zoom)

| Split | Images |
|---|---|
| Train | 18,741 |
| Validation | 4,014 |
| Test | 4,022 |

---

## 4. Model — EfficientNetB0 (Transfer Learning)

Instead of training a CNN from scratch (needs huge data + compute), an
ImageNet-pretrained EfficientNetB0 backbone is reused and adapted to the 10 classes.

```
Input (128×128×3)
  → Data Augmentation (train only)
  → EfficientNetB0 base  (frozen, ImageNet weights)
  → GlobalAveragePooling2D
  → Dropout (0.2)
  → Dense(10, softmax)
```

| Setting | Value |
|---|---|
| Backbone | EfficientNetB0 |
| Optimizer | Adam (lr 1e-3) |
| Loss | Categorical cross-entropy |
| Epochs | 12 (frozen) + fine-tuning (top layers, lr 1e-5) |
| Hardware | CPU |

A MobileNetV2 version (`train_model.py`) is also included for comparison.
After the frozen stage, `fine_tune.py` unfreezes the top layers of the base and
re-trains at a very low learning rate, lifting accuracy 94.2% → 96.1%.

---

## 5. Results

Two training stages were run:

| Stage | Test Accuracy |
|---|---|
| Frozen transfer learning (head only) | 94.21% |
| **+ Fine-tuning (top layers unfrozen)** | **96.10%** |

**Final: 96.1% test accuracy** · macro-F1 **0.960** · weighted-F1 **0.961**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | 0.940 | 0.965 | 0.952 |
| Forest | 0.976 | 0.991 | 0.983 |
| HerbaceousVegetation | 0.953 | 0.944 | 0.949 |
| Highway | 0.934 | 0.949 | 0.942 |
| Industrial | 0.984 | 0.973 | 0.979 |
| Pasture | 0.956 | 0.940 | 0.948 |
| PermanentCrop | 0.917 | 0.912 | 0.915 |
| Residential | 0.974 | 0.998 | 0.986 |
| River | 0.975 | 0.941 | 0.958 |
| SeaLake | 0.998 | 0.979 | 0.988 |

**Observations**

- Every class reaches F1 ≥ 0.91 — no weak class.
- Strongest: SeaLake, Residential, Forest, Industrial (F1 ≥ 0.98).
- Hardest: **PermanentCrop** (F1 0.915) — crop-type classes (AnnualCrop /
  PermanentCrop / HerbaceousVegetation) look similar from a top-down view.
- Fine-tuning lifted accuracy 94.2% → 96.1% with no overfitting (train/val stayed close).

**Comparison with a published reference**

| Study | Task | Best Model | Test Accuracy |
|---|---|---|---|
| Abd Zaid et al. (2024) | 4 classes | MobileNetV2 (frozen) | 96.1% |
| Abd Zaid et al. (2024) | 4 classes | VGG16 (frozen) | 90.6% |
| Abd Zaid et al. (2024) | 4 classes | Custom CNN | 87.2% |
| **This project** | **10 classes** | **EfficientNetB0 + fine-tuning** | **96.1%** |

This project matches the reference paper's best accuracy on a **harder 10-class task**
(vs. their 4 classes), and adds two techniques the paper did not use:
**fine-tuning** and **perceptual-hash data cleaning**.

**Plots & reports** (in `reports/`):

- `training_curves_finetuned.png` — accuracy & loss curves
- `confusion_matrix_finetuned.png` — per-class confusion matrix
- `classification_report_finetuned.txt` — full metrics
- `sample_prediction_effnet.png` — example prediction
- `class_distribution.png`, `sample_grid.png` — EDA

---

## 6. Project Structure

```
eurosat-cnn/
├── data/
│   ├── raw/EuroSAT_RGB/        # original 27,000 images (10 class folders)
│   └── processed/              # cleaned + resized, split:
│       ├── train/  val/  test/ #   <10 class folders each>
├── reports/                    # charts, cleaning log, metrics
├── data_prep.py                # cleaning + preprocessing + EDA + split
├── train_efficientnet.py       # EfficientNetB0 training + evaluation
├── train_model.py              # MobileNetV2 version (comparison)
├── fine_tune.py                # fine-tuning stage (94.2% -> 96.1%)
├── predict.py                  # test the model on your own images
├── app.py                      # Gradio web app (Hugging Face Spaces demo)
├── eurosat_efficientnetb0_finetuned.keras  # final trained model
├── requirements.txt
└── README.md
```

---

## 7. How to Run

```bash
# 1. create + activate virtual environment (Python 3.12)
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1            # Windows PowerShell

# 2. install dependencies
pip install -r requirements.txt

# 3. clean + preprocess the data
python data_prep.py

# 4. train + evaluate
python train_efficientnet.py

# 5. fine-tune to boost accuracy (94.2% -> 96.1%)
python fine_tune.py

# 6. (optional) test the model on your own satellite images
python predict.py

# 7. (optional) run the web app locally
python app.py
```

---

## 8. Tech Stack

Python 3.12 · TensorFlow / Keras 2.21 · EfficientNetB0 ·
NumPy · pandas · scikit-learn · matplotlib · Pillow · Gradio · Hugging Face Spaces

---

*Built as a CNN image-classification project: data cleaning → preprocessing → EDA →
transfer-learning training → fine-tuning → evaluation → live deployment, achieving
96.1% test accuracy on the 10-class EuroSAT dataset.*

**🔗 Live demo:** https://huggingface.co/spaces/Afeefarehan/eurosat-land-classifier
