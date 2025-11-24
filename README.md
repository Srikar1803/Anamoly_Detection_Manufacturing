# Automated Anomaly Detection in Manufacturing

This project implements a deep-learning–based visual inspection system using the **MVTec AD** dataset.  
Per–category U-Net autoencoders are trained to reconstruct *normal* images; **reconstruction error** is then used to detect and localize surface defects on industrial parts.

The system includes:

- A full training & evaluation pipeline (per category)
- Threshold selection using validation data (max F1-score)
- ROC / AUC and per-class metrics
- An interactive **Gradio** interface for real-time testing and heatmap visualization

---

## 1. Project Overview

**Goal.** Detect and localize visual anomalies (scratches, dents, misprints, texture defects) in manufacturing images.

**Pipeline (high level)**

1. **Data preparation** – load MVTec AD, separate train (good only) vs. test (good + anomaly), compute basic statistics.
2. **Model training** – train a U-Net autoencoder per category (bottle, cable, capsule, …).
3. **Anomaly scoring** – use reconstruction error with a Top-K loss to focus on the most abnormal pixels.
4. **Thresholding** – choose a per-category decision threshold by maximizing F1 on validation data.
5. **Evaluation** – compute AUROC, F1, precision, recall; generate ROC curves and loss curves.
6. **Interface** – Gradio app where a user selects a category, uploads an image, and sees:
   - prediction: **OK vs ANOMALY**
   - anomaly score vs. threshold
   - heatmap overlay highlighting defect regions.

---

## 2. Repository Structure

```text
Anamoly_Detection_Manufacturing/
│
├── Docs/                 # Architecture diagrams, report PDFs, interface screenshots
├── Notebooks/            # Jupyter notebooks (EDA, training, evaluation)
│   ├── Data_analysis.ipynb
│   └── main.ipynb
├── Results/              # Saved plots (loss curves, ROC curves, metric tables)
├── Runs/                 # Per-category model runs, checkpoints, thresholds, predictions
├── User Interface/       # Gradio app / UI code
├── data/                 # (Optional) MVTec AD data or symlink/README on where to download
├── src/                  # Core training / evaluation code (models, dataloaders, utils)
│   ├── models/
│   ├── datasets/
    ├── requirements.txt
│   └── ...
└── README.md
