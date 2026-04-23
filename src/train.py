"""
train.py — Train and compare classifiers for fake news detection.

This module can be run in two ways:

1. Train a SPECIFIC model:
       python3 src/train.py --model logistic_regression
       python3 src/train.py --model naive_bayes
       python3 src/train.py --model svm
       python3 src/train.py --model random_forest
       python3 src/train.py --model gradient_boosting

2. Train ALL models and compare:
       python3 src/train.py --model all

Each trained model is saved individually to models/<model_name>.joblib.
The best model (when running all) is also saved as models/best_pipeline.joblib.
Results are saved to models/results.pkl for use by evaluate.py.
"""

import os
import sys
import time
import pickle
import argparse
import joblib
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_dataset
from src.preprocessing import TextPreprocessor
from src.feature_engineering import get_tfidf_vectorizer


# ── Directories ────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLIT_PATH = os.path.join(DATA_DIR, "train_test_split.pkl")
RESULTS_PATH = os.path.join(MODELS_DIR, "results.pkl")


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Convert sparse matrices to dense arrays.
    Needed for classifiers like HistGradientBoosting that
    don't accept sparse input from TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if issparse(X):
            return X.toarray()
        return X


# ── Model Registry ─────────────────────────────────────────
# Maps CLI names to (display_name, classifier_instance, needs_dense)
MODEL_REGISTRY = {
    "logistic_regression": (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
        False,
    ),
    "naive_bayes": (
        "Multinomial Naive Bayes",
        MultinomialNB(alpha=0.1),
        False,
    ),
    "svm": (
        "Linear SVM (SGD)",
        SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, max_iter=1000, random_state=42),
        False,
    ),
    "random_forest": (
        "Random Forest",
        RandomForestClassifier(n_estimators=200, max_depth=50, n_jobs=-1, random_state=42),
        False,
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        HistGradientBoostingClassifier(max_iter=200, max_depth=10, learning_rate=0.1, random_state=42),
        True,  # needs dense input
    ),
}


def build_pipeline(classifier, needs_dense=False):
    """
    Build a complete sklearn Pipeline for a given classifier.

    Pipeline stages:
    1. TextPreprocessor: clean raw text (lowercase, remove noise, lemmatize)
    2. TfidfVectorizer: convert cleaned text to TF-IDF feature vectors
    3. (Optional) DenseTransformer: for classifiers that need dense input
    4. Classifier: the actual ML model

    Args:
        classifier: sklearn classifier instance
        needs_dense: if True, add DenseTransformer + use smaller TF-IDF

    Returns:
        sklearn.pipeline.Pipeline
    """
    if needs_dense:
        # Use smaller TF-IDF (10K features) to avoid memory issues with dense conversion
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        steps = [
            ("preprocessor", TextPreprocessor()),
            ("tfidf", tfidf),
            ("to_dense", DenseTransformer()),
            ("classifier", classifier),
        ]
    else:
        steps = [
            ("preprocessor", TextPreprocessor()),
            ("tfidf", get_tfidf_vectorizer()),
            ("classifier", classifier),
        ]

    return Pipeline(steps)


def get_train_test_split():
    """
    Load or create the train-test split.

    The split is saved to disk so every model trains/tests on
    exactly the same data — ensuring fair comparison.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    if os.path.exists(SPLIT_PATH):
        print(f"✓ Loading existing train-test split from {SPLIT_PATH}")
        with open(SPLIT_PATH, "rb") as f:
            split = pickle.load(f)
        print(f"  Training: {len(split['X_train']):,}  |  Test: {len(split['X_test']):,}")
        return split["X_train"], split["X_test"], split["y_train"], split["y_test"]

    # Load data and create split
    df = load_dataset()
    X = df["content"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save split to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SPLIT_PATH, "wb") as f:
        pickle.dump({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }, f)

    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set:     {len(X_test):,} samples")
    print(f"✓ Split saved to {SPLIT_PATH}")

    return X_train, X_test, y_train, y_test


def train_single_model(model_key, X_train, X_test, y_train, y_test):
    """
    Train a single model by its registry key.

    Args:
        model_key: key from MODEL_REGISTRY (e.g., "svm")
        X_train, X_test: raw text arrays
        y_train, y_test: label arrays

    Returns:
        dict with model results (accuracy, f1, auc, time, pipeline, y_pred, y_proba)
    """
    display_name, classifier, needs_dense = MODEL_REGISTRY[model_key]

    print(f"\n{'─' * 50}")
    print(f"Training: {display_name}")
    print(f"{'─' * 50}")

    pipeline = build_pipeline(classifier, needs_dense=needs_dense)

    # Train
    start_time = time.time()
    
    with mlflow.start_run(run_name=display_name):
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predict
        y_pred = pipeline.predict(X_test)

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # AUC-ROC
        proba_scores = None
        auc = None
        try:
            proba_scores = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba_scores)
        except AttributeError:
            try:
                proba_scores = pipeline.decision_function(X_test)
                auc = roc_auc_score(y_test, proba_scores)
            except Exception:
                pass
                
        # Log to MLflow
        mlflow.log_param("model_type", display_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        if auc is not None:
            mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("train_time", train_time)
        
        # Save model to MLflow (for production reference)
        mlflow.sklearn.log_model(pipeline, "model")

    result = {
        "display_name": display_name,
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "time": train_time,
        "pipeline": pipeline,
        "y_pred": y_pred,
        "y_proba": proba_scores,
    }

    # Save individual model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{model_key}_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    print(f"  Accuracy:   {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}" if auc else "  AUC-ROC:    N/A")
    print(f"  Time:       {train_time:.1f}s")
    print(f"  ✓ Saved to {model_path}")

    return result


def train_and_compare(model_keys=None):
    """
    Train one or more models and return results.

    Args:
        model_keys: list of model keys to train. If None, trains all.

    Returns:
        tuple: (all_results_dict, best_model_name, best_pipeline, X_test, y_test)
    """
    if model_keys is None:
        model_keys = list(MODEL_REGISTRY.keys())

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Fake_News_Classical_Models")

    # ── Load / create split ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Data & Split")
    print("=" * 60)
    X_train, X_test, y_train, y_test = get_train_test_split()

    # ── Train each model ───────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Training {len(model_keys)} Model(s)")
    print("=" * 60)

    results = {}
    for key in model_keys:
        if key not in MODEL_REGISTRY:
            print(f"\n  ⚠ Unknown model: {key}. Skipping.")
            continue
        display_name = MODEL_REGISTRY[key][0]
        result = train_single_model(key, X_train, X_test, y_train, y_test)
        results[display_name] = result

    # ── Load any previously saved results and merge ────────────
    all_results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            all_results = pickle.load(f)

    all_results.update(results)

    # Save merged results
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(all_results, f)

    # ── Select best model ──────────────────────────────────────
    if all_results:
        best_name = max(all_results, key=lambda k: all_results[k]["f1"])
        best_pipeline = all_results[best_name]["pipeline"]

        print(f"\n{'=' * 60}")
        print(f"Best Model (across all trained): {best_name}")
        print(f"{'=' * 60}")
        print(f"  F1 (macro): {all_results[best_name]['f1']:.4f}")
        print(f"  Accuracy:   {all_results[best_name]['accuracy']:.4f}")

        # Save best pipeline
        best_path = os.path.join(MODELS_DIR, "best_pipeline.joblib")
        joblib.dump(best_pipeline, best_path)
        print(f"  ✓ Best pipeline saved to {best_path}")
    else:
        best_name = None
        best_pipeline = None

    # Print classification report for newly trained models
    for name, r in results.items():
        print(f"\n{'─' * 50}")
        print(f"Classification Report — {name}")
        print(f"{'─' * 50}")
        print(classification_report(
            y_test, r["y_pred"], target_names=["Real", "Fake"],
        ))

    return all_results, best_name, best_pipeline, X_test, y_test


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train fake news detection models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help=(
            "Which model to train. Options:\n"
            "  logistic_regression  — Logistic Regression\n"
            "  naive_bayes          — Multinomial Naive Bayes\n"
            "  svm                  — Linear SVM (SGDClassifier)\n"
            "  random_forest        — Random Forest\n"
            "  gradient_boosting    — Gradient Boosting (HistGradientBoosting)\n"
            "  all                  — Train all 5 models and compare"
        ),
    )
    args = parser.parse_args()

    model_arg = args.model.lower().strip()

    if model_arg == "all":
        keys = None  # train all
    elif model_arg in MODEL_REGISTRY:
        keys = [model_arg]
    else:
        print(f"✗ Unknown model: '{model_arg}'")
        print(f"  Available: {', '.join(MODEL_REGISTRY.keys())}, all")
        sys.exit(1)

    results, best_name, _, _, _ = train_and_compare(model_keys=keys)

    # Print summary table
    if results:
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Model':<30} {'Accuracy':>10} {'F1':>10} {'AUC':>10} {'Time':>10}")
        print("─" * 70)
        for name, r in results.items():
            auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
            marker = " ★" if name == best_name else ""
            print(f"{name:<30} {r['accuracy']:>10.4f} {r['f1']:>10.4f} {auc_str:>10} {r['time']:>9.1f}s{marker}")
