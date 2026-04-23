"""
evaluate.py — Generate evaluation reports and visualizations.

This module can be run independently to generate reports based on
whichever models have been trained so far.

Run independently:
    python3 src/evaluate.py

Produces:
- classification_report.txt: Per-class precision, recall, F1
- confusion_matrix.png: Heatmap of TP/FP/TN/FN
- roc_curve.png: ROC curve with AUC score for the best model
- model_comparison.png: Bar chart comparing all trained models
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import get_train_test_split, RESULTS_PATH, DenseTransformer

# Output directory for reports
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


def setup_plot_style():
    """Configure matplotlib for clean, professional plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
    })


def save_classification_report(y_test, y_pred, best_name):
    """Save a detailed classification report as a text file."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = classification_report(
        y_test, y_pred, target_names=["Fake", "Real"], digits=4
    )

    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Classification Report — {best_name}\n")
        f.write("=" * 55 + "\n\n")
        f.write(report)

    print(f"  ✓ Classification report saved to {report_path}")


def plot_confusion_matrix(y_test, y_pred, best_name):
    """Generate and save a confusion matrix heatmap."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    setup_plot_style()

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax,
        cbar_kws={"label": "Count"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {best_name}")

    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Confusion matrix saved to {path}")


def plot_roc_curve(y_test, best_proba_scores, best_name):
    """Generate and save an ROC curve with AUC score."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    setup_plot_style()

    if best_proba_scores is None:
        print("  ⚠ ROC curve not available (no probability output for best model)")
        return

    fpr, tpr, thresholds = roc_curve(y_test, best_proba_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2.5, label=f"{best_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {best_name}")
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ ROC curve saved to {path}")


def plot_model_comparison(results):
    """Generate and save a bar chart comparing all trained models."""
    if not results:
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    setup_plot_style()

    model_names = list(results.keys())
    # Shorten names for display
    short_names = [n.replace("Multinomial ", "").replace(" (SGD)", "") for n in model_names]

    accuracies = [results[n]["accuracy"] for n in model_names]
    f1_scores = [results[n]["f1"] for n in model_names]
    auc_scores = [results[n]["auc"] if results[n]["auc"] else 0 for n in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy", color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x, f1_scores, width, label="F1 (macro)", color="#2196F3", alpha=0.85)
    bars3 = ax.bar(x + width, auc_scores, width, label="AUC-ROC", color="#FF9800", alpha=0.85)

    # Add value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.005,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Fake News Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.3, label="0.90 threshold")

    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, "model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Model comparison chart saved to {path}")


def generate_all_reports():
    """Load results and generate all evaluation reports."""
    print("\n" + "=" * 60)
    print("Generating Evaluation Reports")
    print("=" * 60)

    if not os.path.exists(RESULTS_PATH):
        print("  ⚠ No results found. Train a model first:")
        print("    python3 src/train.py --model all")
        return

    with open(RESULTS_PATH, "rb") as f:
        results = pickle.load(f)

    if not results:
        print("  ⚠ Results file is empty.")
        return

    # Load test labels
    _, _, _, y_test = get_train_test_split()

    best_name = max(results, key=lambda k: results[k]["f1"])
    y_pred = results[best_name]["y_pred"]
    y_proba = results[best_name]["y_proba"]

    print(f"  Generating reports based on {len(results)} trained models...")
    print(f"  Best model: {best_name}")

    save_classification_report(y_test, y_pred, best_name)
    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_roc_curve(y_test, y_proba, best_name)
    plot_model_comparison(results)

    print(f"\n  ✓ All reports saved to {REPORTS_DIR}/")


if __name__ == "__main__":
    generate_all_reports()
