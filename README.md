# MSCS 633 – Assignment 4: Fraud Detection with AutoEncoder (PyOD)

A deep-learning fraud-detection pipeline built with the
[PyOD](https://pyod.readthedocs.io/) AutoEncoder on the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
dataset.

---

## 1. Project Overview

AutoEncoders learn a compressed representation of normal (legitimate)
transactions. When a fraudulent transaction passes through the network, its
reconstruction error is significantly higher than that of normal samples.
PyOD wraps this idea in a clean scikit-learn-style API, making it straightforward to:

- Train on raw (scaled) feature vectors.
- Obtain a per-sample anomaly score via `decision_function`.
- Predict binary inlier / outlier labels via `predict`.
- Estimate outlier probabilities via `predict_proba`.

---

## 2. Repository Structure

```
MSCS633_Assignment4/
├── fraud_detection.py   # Main script – runs the full pipeline
├── creditcard.csv       # Dataset
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── figures/             # Auto-created at runtime; stores all output plots
    ├── class_distribution.png
    ├── amount_distribution.png
    ├── correlation_heatmap.png
    ├── reconstruction_error.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── precision_recall_curve.png
```

---

## 3. Dataset Setup

The credit card fraud dataset is **not included in the repository** due to its size.
You must download and unzip it manually before running the script.

1. **Download** `creditcard.csv.zip` from the course submission / shared link
   (or directly from
   [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).
2. **Unzip** the archive:

   ```bash
   unzip creditcard.csv.zip
   ```

3. **Place `creditcard.csv`** in the repository root so the directory looks like:

   ```
   MSCS633_Assignment4/
   ├── creditcard.csv       ← required here
   ├── fraud_detection.py
   ├── requirements.txt
   └── README.md
   ```

> If `creditcard.csv` is missing when you run the script, it will exit immediately
> with a clear error message pointing you back to this section.

---

## 4. Prerequisites  

| Requirement | Version                |
|-------------|------------------------|
| Python      | 3.9 – 3.11 recommended |
| pip         | >= 22                  |

> **PyTorch note** – PyOD 2.x's `AutoEncoder` uses **PyTorch** as its backend.
> PyTorch supports **macOS (CPU/MPS)**, **Linux**, and **Windows**.
> Apple-Silicon (M-series) users automatically benefit from MPS acceleration
> when `torch` detects it; no extra packages are required.

---

## 5. Installation

### 5.1 Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate.bat   # Windows
```

### 5.2 Upgrade pip

```bash
pip3 install --upgrade pip
```

### 5.3 Install all dependencies

```bash
pip3 install -r requirements.txt
```

This installs:

| Package        | Purpose                                              |
|----------------|------------------------------------------------------|
| `pyod`         | Outlier / anomaly detection library (AutoEncoder)    |
| `torch`        | Deep-learning backend used by PyOD 2.x's AutoEncoder |
| `tqdm`         | Progress-bar dependency required by PyOD             |
| `numpy`        | Numerical computing                                  |
| `pandas`       | Data manipulation                                    |
| `matplotlib`   | Plotting                                             |
| `seaborn`      | Statistical visualisation                            |
| `scikit-learn` | Pre-processing, metrics, train/test split            |

---

## 6. Running the Script

```bash
python3 fraud_detection.py
```

The full pipeline takes **2 – 5 minutes** depending on hardware
(PyTorch training loop dominates the runtime).

---

## 7. Pipeline Steps

| Step | Function              | Description                                                                   |
|------|-----------------------|-------------------------------------------------------------------------------|
| 1    | `load_dataset`        | Reads `creditcard.csv` from the repository directory                          |
| 2    | `run_eda`             | Prints dataset info, class counts, missing values; saves 3 plots              |
| 3    | `preprocess`          | Drops `Time`, scales with `StandardScaler`, stratified 80/20 split            |
| 4    | `build_autoencoder`   | Creates the PyOD `AutoEncoder` with all hyperparameters explicitly set        |
| 4    | `train_model`         | Fits the AutoEncoder on mostly-legitimate training data                       |
| 5    | `evaluate_model`      | Saves 4 evaluation plots; prints classification report and key metrics        |
| 6    | `demonstrate_pyod_features` | Explicitly calls every major PyOD AutoEncoder API method with annotations |

---

## 8. Output

### Console output includes

- Dataset shape, dtypes, descriptive statistics
- Class distribution and fraud percentage
- Missing-value counts
- PyTorch training progress log (50 epochs)
- Full classification report (precision, recall, F1 per class)
- ROC-AUC, Average Precision, and F1 scores
- Detailed output for each PyOD API method

### Figures saved to `./figures/`

| File                         | Content                                                     |
|------------------------------|-------------------------------------------------------------|
| `class_distribution.png`     | Bar chart – number of legit vs. fraud transactions          |
| `amount_distribution.png`    | Transaction-amount histograms split by class                |
| `correlation_heatmap.png`    | Pearson correlations for V1–V10 + Amount                    |
| `reconstruction_error.png`   | Overlapping histograms of anomaly scores by true class      |
| `confusion_matrix.png`       | Heat-map of true vs. predicted labels                       |
| `roc_curve.png`              | ROC curve with AUC annotation                               |
| `precision_recall_curve.png` | Precision-Recall curve with Average Precision annotation    |