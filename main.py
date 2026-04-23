"""
main.py — Modular CLI entry point for the Fake News Detector.
"""

import os
import sys
import argparse
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# We must import DenseTransformer and HuggingFaceEmbedder into main namespace 
# so that joblib can unpickle the pipelines globally if needed by other scripts.
from src.train import DenseTransformer
from src.train_hf import HuggingFaceEmbedder

from src.data_loader import load_dataset
from src.train import train_and_compare
from src.train_hf import train_hf_model
from src.evaluate import generate_all_reports

def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detector Production Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["data", "train_classical", "train_hf", "eval", "api", "all_classical"],
        help=(
            "Which pipeline step to execute:\n"
            "  data            - Download and preprocess the dataset\n"
            "  train_classical - Train classical ML models (TF-IDF + Sklearn)\n"
            "  train_hf        - Train Hugging Face model (SentenceTransformers)\n"
            "  eval            - Generate evaluation reports & plots\n"
            "  api             - Start the FastAPI server\n"
            "  all_classical   - Run data, train_classical, and eval sequentially"
        )
    )
    
    args = parser.parse_args()
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        FAKE NEWS DETECTOR — PRODUCTION PIPELINE          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Executing step: {args.step.upper()}\n")

    if args.step == "data":
        load_dataset(force_reload=True)
        
    elif args.step == "train_classical":
        # Ensure data exists first
        load_dataset()
        train_and_compare(model_keys=None)
        
    elif args.step == "train_hf":
        # Ensure data exists first
        load_dataset()
        train_hf_model()
        
    elif args.step == "eval":
        generate_all_reports()
        
    elif args.step == "all_classical":
        load_dataset()
        train_and_compare()
        generate_all_reports()
        
    elif args.step == "api":
        print("Starting FastAPI server...")
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
