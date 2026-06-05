# 🫁 Deep Learning Based Pneumonia Detection
## with NLP-Driven Clinical Explanation System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 📋 Project Overview

A production-grade AI system that:

1. **Classifies chest X-rays** as Normal or Pneumonia (binary) or Normal / Bacterial / Viral (multiclass) using a fine-tuned **EfficientNetB3** backbone.
2. **Explains predictions visually** via **Grad-CAM** heatmaps highlighting the discriminative lung regions.
3. **Quantifies uncertainty** using **Monte Carlo Dropout** (epistemic uncertainty over N stochastic forward passes).
4. **Generates clinical reports** via an **NLP Explanation Engine** combining rule-based medical language templates with confidence- and uncertainty-aware text selection.

---

## 📁 Project Structure

```
pneumonia_detection/
├── notebooks/
│   ├── pneumonia_detection_main.ipynb   ← Main Kaggle notebook (full pipeline)
│   └── multiclass_subtype.ipynb         ← Bacterial vs Viral classification
├── src/
│   ├── model.py        ← PneumoniaNet (EfficientNetB3 + MC Dropout + Grad-CAM)
│   ├── dataset.py      ← ChestXRayDataset, MultiClassXRayDataset, transforms
│   ├── trainer.py      ← Trainer class, training loops, metrics
│   └── nlp_engine.py   ← ClinicalExplanationEngine (NLP report generation)
└── README.md
```

---

## 🗂️ Dataset

### Primary Dataset
**Chest X-Ray Images (Pneumonia)**  
📦 [kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

The PNEUMONIA folder contains files prefixed `bacteria_*` and `virus_*`, enabling 3-class subtype classification.

### Optional Additional Datasets (for experiments)
| Dataset | Link | Use |
|---------|------|-----|
| NIH ChestX-ray14 | [kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data) | 14 thoracic diseases, larger scale |
| CheXpert | [stanfordmlgroup.github.io/competitions/chexpert](https://stanfordmlgroup.github.io/competitions/chexpert/) | Stanford dataset with uncertainty labels |
| RSNA Pneumonia Detection | [kaggle.com/competitions/rsna-pneumonia-detection-challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) | Detection + localisation |

---

## 🚀 Running on Kaggle

### Step 1 — Add the dataset
1. Open your Kaggle notebook
2. Click **"+ Add Data"** → search **"chest-xray-pneumonia"**
3. Add dataset by **Paul Mooney**

### Step 2 — Run the main notebook
Open `notebooks/pneumonia_detection_main.ipynb` and run all cells.

> Enable **GPU** in Notebook Settings → Accelerator → **GPU T4 x2**

---

## 🏗️ Architecture Details

### PneumoniaNet
```
EfficientNetB3 Backbone (pretrained ImageNet)
    └─ conv_head  [Grad-CAM target layer]
         └─ AdaptiveAvgPool2d(1)
              └─ Flatten
                   └─ BN → Dropout(0.3) → Linear(1536→512) → GELU
                        └─ BN → Dropout(0.3) → Linear(512→128) → GELU
                             └─ Dropout(0.15) → Linear(128→num_classes)
```

### Grad-CAM
- Registers forward & backward hooks on `backbone.conv_head`
- Computes gradient-weighted spatial importance map
- Bicubic upsampling to 224×224
- Overlaid on original X-ray with JET colormap

### Monte Carlo Dropout
- During inference, dropout layers remain active (`.train()` mode)
- N=30 stochastic forward passes per image
- Mean prediction + std (uncertainty) computed across passes
- Predictive entropy = `−∑ p log p` over mean class probabilities

### NLP Clinical Report Engine
- Confidence-tier classification: HIGH (>85%), MID (65–85%), LOW (<65%)
- Uncertainty-tier classification: LOW (<5% std), MEDIUM (5–15%), HIGH (>15%)
- Template selection based on diagnosis × confidence × uncertainty
- Structured output: study info, findings, recommendations, disclaimer

---

## 📊 Expected Results

| Metric           | Binary | Multiclass |
|------------------|--------|------------|
| Test Accuracy    | ~95%   | ~88%       |
| ROC-AUC          | ~0.98  | ~0.95 (OvR)|
| Sensitivity      | ~97%   | —          |
| Specificity      | ~89%   | —          |

---

## 📈 Output Files (in `/kaggle/working/`)

| File | Description |
|------|-------------|
| `best_model.pth` | Binary classifier weights |
| `best_model3.pth` | Multiclass classifier weights |
| `training_curves.png` | Loss / Accuracy / AUC curves |
| `evaluation_metrics.png` | Confusion matrix + ROC curve |
| `gradcam_visualization.png` | Grad-CAM overlays on test samples |
| `mc_dropout_uncertainty.png` | MC Dropout probability distribution |
| `inference_output.png` | Full pipeline inference display |
| `test_predictions.csv` | All test predictions with confidence |

---

## 🔧 Key Dependencies

```
torch>=2.0          # Deep learning framework
torchvision>=0.15   # Image transforms
timm>=0.9           # EfficientNet pretrained models
scikit-learn        # Metrics
opencv-python       # Grad-CAM heatmap overlay
matplotlib          # Visualizations
seaborn             # Confusion matrix plots
pandas / numpy      # Data handling
tqdm                # Progress bars
```

---

## 💡 Extending the Project

- **Replace EfficientNetB3** with DenseNet121, ViT-B/16, or ConvNeXt for comparison
- **Swap NLP engine** with a fine-tuned clinical language model (e.g., BioGPT, ClinicalBERT) for more natural language generation
- **Add segmentation head** (U-Net branch) for lung region segmentation before classification
- **Deploy as FastAPI service** — wrap `full_inference_pipeline()` in a REST endpoint

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.  
It is **not** a certified medical device and should **not** be used for clinical diagnosis.  
Always consult qualified medical professionals for healthcare decisions.

---

*Built with EfficientNetB3, Grad-CAM, Monte Carlo Dropout, and NLP-driven clinical explanations.*
