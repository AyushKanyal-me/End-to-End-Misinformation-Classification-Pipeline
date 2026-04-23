import sys
import os
import joblib
import gradio as gr

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# We must import custom classes so joblib can unpickle the model
try:
    from src.train import DenseTransformer
    from src.train_hf import HuggingFaceEmbedder
    setattr(sys.modules['__main__'], 'DenseTransformer', DenseTransformer)
    setattr(sys.modules['__main__'], 'HuggingFaceEmbedder', HuggingFaceEmbedder)
except ImportError:
    pass

# Load the best model (check both local 'models/' folder and root folder for Hugging Face)
MODEL_PATHS = ["models/best_pipeline.joblib", "best_pipeline.joblib"]
pipeline = None

for path in MODEL_PATHS:
    if os.path.exists(path):
        print(f"Loading model from {path}...")
        pipeline = joblib.load(path)
        break

def classify_text(text):
    if not pipeline:
        return "Error: Model not found. Please train the model first.", 0.0
    
    if len(text.strip()) < 20:
        return "Please enter a longer news excerpt (at least 20 characters).", 0.0
        
    try:
        prediction = pipeline.predict([text])[0]
        # WELFake dataset: label 0 = Real, label 1 = Fake
        label = "🔴 Fake News" if prediction == 1 else "🟢 Real News"
        
        try:
            proba = pipeline.predict_proba([text])[0]
            confidence = max(proba)
        except:
            confidence = 1.0
            
        return label, f"{confidence:.1%}"
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️ Misinformation Classification System")
    gr.Markdown("Paste a news article or headline below to determine if it is Real or Fake using our NLP pipeline.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=8, 
                placeholder="Paste news article text here...", 
                label="Article Text"
            )
            submit_btn = gr.Button("Analyze Article", variant="primary")
            
        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Classification")
            output_conf = gr.Textbox(label="Confidence Score")
            
    submit_btn.click(
        fn=classify_text,
        inputs=input_text,
        outputs=[output_label, output_conf]
    )

if __name__ == "__main__":
    print("Starting Web UI on http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860)
