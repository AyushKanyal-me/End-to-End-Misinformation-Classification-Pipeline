import os
import sys
import joblib

# Ensure the project root is in the python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import DenseTransformer so joblib can successfully unpickle pipelines that use it
from src.train import DenseTransformer

def check_custom_news():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_pipeline.joblib')
    
    if not os.path.exists(model_path):
        print(f"⚠ Model not found at {model_path}")
        print("Please run `python main.py` first to train the models.")
        return

    print("Loading the best model...")
    pipeline = joblib.load(model_path)
    
    print("\n" + "=" * 60)
    print("🕵️ Fake News Detector - Interactive Mode")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop.")

    while True:
        print("\n" + "-" * 60)
        try:
            # Get multi-line input or just a single long string
            print("Paste the news article text below and press Enter:")
            text = input("> ")
            
            if text.strip().lower() in ['quit', 'exit']:
                print("Exiting...")
                break
                
            if not text.strip():
                continue
                
            # Make prediction
            prediction = pipeline.predict([text])[0]
            label = "✅ REAL" if prediction == 1 else "❌ FAKE"
            
            # Get confidence score if the model supports it
            try:
                proba = pipeline.predict_proba([text])[0]
                confidence = max(proba) * 100
                conf_str = f" ({confidence:.1f}% confidence)"
            except AttributeError:
                conf_str = ""
                
            print("\nRESULT:")
            print(f"  --> {label}{conf_str}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except EOFError:
            break

if __name__ == "__main__":
    check_custom_news()
