"""
This script:
  1. Loads the anonymized credit card transactions dataset from creditcard.csv
     (included in the repository).
  2. Performs exploratory data analysis (EDA) and preprocessing.
  3. Trains a PyOD AutoEncoder model to detect fraudulent transactions.
  4. Evaluates the model with multiple metrics and visualisations.
  5. Demonstrates all key PyOD AutoEncoder features required by the assignment.

Dataset  : https://www.kaggle.com/datasets/whenamancodes/fraud-detection
"""

# ── Standard library ────────────────────────────────────────────────────────
import os
import sys
import warnings

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

from pyod.models.auto_encoder import AutoEncoder

warnings.filterwarnings("ignore")

# ── Global settings ──────────────────────────────────────────────────────────
RANDOM_STATE  = 42
CONTAMINATION = 0.001   # ~0.1 % fraud rate mirrors the dataset imbalance
FIGURES_DIR   = "figures"
DATASET_CSV   = "creditcard.csv"   # bundled with the repository

os.makedirs(FIGURES_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA ACQUISITION
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """
    Load the credit card fraud CSV dataset bundled with the repository.

    The file `creditcard.csv` must be present in the same directory as this
    script.  It is included in the repository and requires no download step.

    Returns
    -------
    pd.DataFrame
        Raw credit card transactions dataset.
    """
    print("=" * 60)
    print("STEP 1 – Loading dataset")
    print("=" * 60)

    if not os.path.isfile(DATASET_CSV):
        print(
            f"\nERROR: '{DATASET_CSV}' not found.\n"
            f"  Expected location: {os.path.abspath(DATASET_CSV)}\n"
            "  Make sure creditcard.csv is in the same directory as fraud_detection.py."
        )
        sys.exit(1)

    print(f"Loading: {os.path.abspath(DATASET_CSV)}")
    df = pd.read_csv(DATASET_CSV)
    print(f"Dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> None:
    """Print summary statistics and save class-distribution and correlation plots."""
    print("\n" + "=" * 60)
    print("STEP 2 – Exploratory Data Analysis")
    print("=" * 60)

    print("\n── Dataset info ──────────────────────────────────────────")
    print(df.info())

    print("\n── First 5 rows ──────────────────────────────────────────")
    print(df.head())

    print("\n── Descriptive statistics ────────────────────────────────")
    print(df.describe())

    print("\n── Missing values ────────────────────────────────────────")
    print(df.isnull().sum())

    # Class distribution
    class_counts = df["Class"].value_counts()
    print("\n── Class distribution ────────────────────────────────────")
    print(class_counts)
    fraud_pct = class_counts[1] / len(df) * 100
    print(f"   Fraud percentage: {fraud_pct:.4f} %")

    # Plot 1 – Class distribution bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts.plot(kind="bar", color=["steelblue", "crimson"], ax=ax)
    ax.set_title("Class Distribution (0 = Legit, 1 = Fraud)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + 0.1, p.get_height() + 100))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"\n   ✓ Saved figures/class_distribution.png")

    # Plot 2 – Transaction amount distribution by class
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[df["Class"] == 0]["Amount"].clip(upper=2500).hist(
        bins=80, ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set_title("Legit – Transaction Amount")
    axes[0].set_xlabel("Amount (USD)")
    df[df["Class"] == 1]["Amount"].hist(
        bins=80, ax=axes[1], color="crimson", edgecolor="white"
    )
    axes[1].set_title("Fraud – Transaction Amount")
    axes[1].set_xlabel("Amount (USD)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "amount_distribution.png"), dpi=150)
    plt.close(fig)
    print("   ✓ Saved figures/amount_distribution.png")

    # Plot 3 – Correlation heat-map (sample of PCA features + Amount)
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    sample_cols = feature_cols[:10] + ["Amount"]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df[sample_cols].corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heat-map (V1–V10 + Amount)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("   ✓ Saved figures/correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Prepare feature matrix and labels.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names                     : list[str]
    """
    print("\n" + "=" * 60)
    print("STEP 3 – Preprocessing")
    print("=" * 60)

    # Drop 'Time' - it carries no useful fraud signal in this dataset
    df = df.drop(columns=["Time"])

    # Separate features and labels
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    feature_names = df.drop(columns=["Class"]).columns.tolist()

    print(f"Feature matrix shape : {X.shape}")
    print(f"Label vector shape   : {y.shape}")

    # Standardise features (AutoEncoder works best with normalised inputs)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("   ✓ Features standardised with StandardScaler")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train size : {X_train.shape[0]:,} | fraud in train : {y_train.sum():,}")
    print(f"   Test  size : {X_test.shape[0]:,}  | fraud in test  : {y_test.sum():,}")

    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL – PyOD AutoEncoder
# ─────────────────────────────────────────────────────────────────────────────

def build_autoencoder() -> AutoEncoder:
    """
    Instantiate a PyOD AutoEncoder with explicit architecture settings.

    Key hyperparameters
    -------------------
    hidden_neurons  : encoder [64, 32] → bottleneck [16] → decoder [32, 64]
    hidden_activation: ReLU activation in hidden layers
    output_activation: Sigmoid activation at the output layer
    loss            : Mean Squared Error – reconstruction loss
    optimizer       : Adam
    epochs          : 50 training epochs
    batch_size      : 256 samples per mini-batch
    dropout_rate    : 0.2 – regularisation to prevent over-fitting
    contamination   : expected proportion of outliers (fraud rate)
    """
    model = AutoEncoder(
        hidden_neurons=[64, 32, 32, 64],  # symmetric encoder-decoder
        hidden_activation="relu",
        output_activation="sigmoid",
        loss="mse",
        optimizer="adam",
        epochs=50,
        batch_size=256,
        dropout_rate=0.2,
        l2_regularizer=0.1,
        validation_size=0.1,
        preprocessing=False,   # we already standardised outside PyOD
        verbose=1,
        random_state=RANDOM_STATE,
        contamination=CONTAMINATION,
    )
    return model


def train_model(model: AutoEncoder, X_train: np.ndarray) -> AutoEncoder:
    """Fit the AutoEncoder on the (mostly legitimate) training data."""
    print("\n" + "=" * 60)
    print("STEP 4 – Training the AutoEncoder")
    print("=" * 60)
    print("Architecture  : 64 → 32 → [bottleneck] → 32 → 64")
    print("Activation    : ReLU (hidden) / Sigmoid (output)")
    print("Loss          : MSE   |  Optimizer : Adam")
    print("Epochs        : 50    |  Batch     : 256")
    print("Dropout       : 0.2   |  L2 reg    : 0.1")
    print("-" * 60)

    model.fit(X_train)
    print("\n   ✓ Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: AutoEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Evaluate the trained AutoEncoder and produce:
      - Reconstruction error histogram
      - Confusion matrix heat-map
      - ROC curve
      - Precision-Recall curve
      - Console classification report
    """
    print("\n" + "=" * 60)
    print("STEP 5 – Evaluation")
    print("=" * 60)

    # ── 5a. Predictions ──────────────────────────────────────────────────────
    # predict() returns binary labels (0 = inlier / 1 = outlier)
    y_pred = model.predict(X_test)

    # decision_function() returns raw anomaly scores (reconstruction errors)
    anomaly_scores = model.decision_function(X_test)

    print("\n── Classification Report ─────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    roc_auc = roc_auc_score(y_test, anomaly_scores)
    avg_prec = average_precision_score(y_test, anomaly_scores)
    f1 = f1_score(y_test, y_pred)

    print(f"ROC-AUC Score          : {roc_auc:.4f}")
    print(f"Average Precision Score: {avg_prec:.4f}")
    print(f"F1-Score (fraud class) : {f1:.4f}")

    # ── 5b. Reconstruction error histogram ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        anomaly_scores[y_test == 0],
        bins=100,
        alpha=0.6,
        color="steelblue",
        label="Legit",
        density=True,
    )
    ax.hist(
        anomaly_scores[y_test == 1],
        bins=100,
        alpha=0.6,
        color="crimson",
        label="Fraud",
        density=True,
    )
    threshold = model.threshold_
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_title("Reconstruction Error Distribution by Class")
    ax.set_xlabel("Anomaly Score (Reconstruction Error)")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "reconstruction_error.png"), dpi=150)
    plt.close(fig)
    print("\n   ✓ Saved figures/reconstruction_error.png")

    # ── 5c. Confusion matrix ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Fraud"],
        yticklabels=["Legit", "Fraud"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("   ✓ Saved figures/confusion_matrix.png")

    # ── 5d. ROC curve ────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)
    print("   ✓ Saved figures/roc_curve.png")

    # ── 5e. Precision-Recall curve ───────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(y_test, anomaly_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2, label=f"AP = {avg_prec:.4f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)
    print("   ✓ Saved figures/precision_recall_curve.png")

    print(f"\n   All figures saved to: ./{FIGURES_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# 6. DEMONSTRATE PyOD AutoEncoder FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_pyod_features(
    model: AutoEncoder,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Explicitly demonstrate the PyOD AutoEncoder API:
      - decision_function  : per-sample anomaly scores
      - predict            : binary labels (0/1)
      - predict_proba      : probability of being an outlier
      - threshold_         : learned decision threshold
      - labels_            : training-set labels (after fit)
    """
    print("\n" + "=" * 60)
    print("STEP 6 – PyOD AutoEncoder Feature Demonstration")
    print("=" * 60)

    # ── decision_function ────────────────────────────────────────────────────
    scores = model.decision_function(X_test)
    print(f"\n[decision_function] Returns reconstruction errors (anomaly scores).")
    print(f"   Min  : {scores.min():.6f}")
    print(f"   Max  : {scores.max():.6f}")
    print(f"   Mean : {scores.mean():.6f}")
    print(f"   Std  : {scores.std():.6f}")

    # ── predict ──────────────────────────────────────────────────────────────
    predictions = model.predict(X_test)
    print(f"\n[predict] Binary labels – 0 = inlier (legit), 1 = outlier (fraud).")
    unique, counts = np.unique(predictions, return_counts=True)
    for label, count in zip(unique, counts):
        tag = "Legit" if label == 0 else "Fraud"
        print(f"   {tag}: {count:,} samples predicted")

    # ── predict_proba ────────────────────────────────────────────────────────
    proba = model.predict_proba(X_test, method="linear")
    print(f"\n[predict_proba] Outlier probabilities (method='linear').")
    print(f"   Sample fraud probabilities (first 10 known-fraud samples):")
    fraud_indices = np.where(y_test == 1)[0]
    for i, idx in enumerate(fraud_indices[:10]):
        print(f"   Sample {idx:5d}: fraud_prob = {proba[idx, 1]:.4f}  | true_label = {y_test[idx]}")

    # ── threshold_ ───────────────────────────────────────────────────────────
    print(f"\n[threshold_] Decision threshold learned from contamination ratio.")
    print(f"   contamination = {CONTAMINATION}")
    print(f"   threshold_    = {model.threshold_:.6f}")
    print(f"   Samples above this score are flagged as fraud.")

    # ── labels_ (training labels assigned by the model) ─────────────────────
    print(f"\n[labels_] Training-set inlier/outlier assignments (0/1).")
    unique_train, counts_train = np.unique(model.labels_, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        tag = "Inlier" if label == 0 else "Outlier"
        print(f"   {tag}: {count:,} training samples")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate all pipeline steps."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  MSCS 633 – Assignment 4: Fraud Detection (AutoEncoder)  ║")
    print("╚" + "═" * 58 + "╝\n")

    # Step 1 – Load dataset
    df = load_dataset()

    # Step 2 – EDA
    run_eda(df)

    # Step 3 – Preprocessing
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)

    # Step 4 – Build & Train AutoEncoder
    model = build_autoencoder()
    model = train_model(model, X_train)

    # Step 5 – Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 6 – Demonstrate PyOD features
    demonstrate_pyod_features(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("Pipeline complete.  All figures saved to ./figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
