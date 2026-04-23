import os
import sys
import time
import joblib
import numpy as np
import mlflow
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.train import get_train_test_split, MODELS_DIR

# Custom transformer for hugging face embeddings
class HuggingFaceEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.model is None:
            # Import here to avoid loading torch if we don't need it
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        # SentenceTransformer can process a list of strings
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return self.model.encode(X, show_progress_bar=False)

def train_hf_model():
    print("\n" + "=" * 60)
    print("Training Hugging Face Model (SentenceTransformers + GBM)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = get_train_test_split()
    
    # Use a small subset for demonstration to prevent OOM and long execution times on laptops
    X_train, y_train = X_train[:1000], y_train[:1000]
    X_test, y_test = X_test[:200], y_test[:200]
    
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Fake_News_HF_Model")
    
    start_time = time.time()
    
    with mlflow.start_run(run_name="HF_SentenceTransformer_GBM"):
        print("Embedding text using SentenceTransformers... (This may take a few minutes)")
        embedder = HuggingFaceEmbedder()
        X_train_emb = embedder.transform(X_train)
        X_test_emb = embedder.transform(X_test)
        
        print("Training Gradient Boosting Classifier on embeddings...")
        classifier = HistGradientBoostingClassifier(random_state=42)
        classifier.fit(X_train_emb, y_train)
        
        train_time = time.time() - start_time
        
        print("Evaluating...")
        y_pred = classifier.predict(X_test_emb)
        proba_scores = classifier.predict_proba(X_test_emb)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(y_test, proba_scores)
        
        mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")
        mlflow.log_param("classifier", "HistGradientBoosting")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("train_time", train_time)
        
        pipeline = Pipeline([
            ("embedder", HuggingFaceEmbedder()),
            ("classifier", classifier)
        ])
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, "hf_pipeline.joblib")
        joblib.dump(pipeline, model_path)
        
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"\n  Accuracy:   {acc:.4f}")
        print(f"  F1 (macro): {f1:.4f}")
        print(f"  AUC-ROC:    {auc:.4f}")
        print(f"  Time:       {train_time:.1f}s")
        print(f"  ✓ Saved to {model_path}")

        print("\n" + classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

if __name__ == "__main__":
    train_hf_model()
