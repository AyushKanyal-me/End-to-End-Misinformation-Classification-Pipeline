"""
Gradio UI for Hugging Face Spaces.
Calls the remote FastAPI backend on Render instead of running the model locally.
Architecture: Client (HF Spaces) → REST API (Render / FastAPI) → ML Model
"""

import os
import gradio as gr
import requests

# The Render API URL — set this as a Secret in your HF Space settings
API_URL = os.environ.get("API_URL", "").rstrip("/")

def classify_text(text):
    if not API_URL:
        return "⚠️ Error: API_URL environment variable is not set. Add it in your HF Space Settings → Secrets.", ""

    if len(text.strip()) < 20:
        return "Please enter a longer news excerpt (at least 20 characters).", ""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            label = data["prediction"]
            confidence = data["confidence"]

            if label == "Fake":
                display = f"🔴 Fake News"
            else:
                display = f"🟢 Real News"

            return display, f"{confidence:.1f}%"

        elif response.status_code == 503:
            return "⚠️ Model is still loading on the server. Please try again in 30 seconds.", ""
        else:
            return f"⚠️ API returned status {response.status_code}: {response.text}", ""

    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach the API server. It may be starting up (free tier cold start). Try again in 30-60 seconds.", ""
    except requests.exceptions.Timeout:
        return "⚠️ API request timed out. The server may be waking up. Try again in a moment.", ""
    except Exception as e:
        return f"⚠️ Error: {str(e)}", ""


# Build the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️ Misinformation Classification System")
    gr.Markdown(
        "Paste a news article below to classify it as Real or Fake.\n\n"
        "**Architecture:** This UI sends your text to a FastAPI backend hosted on Render, "
        "which runs the ML model and returns the prediction."
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=8,
                placeholder="Paste a full news article here for best results...",
                label="Article Text",
            )
            submit_btn = gr.Button("🔍 Analyze Article", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Textbox(label="Classification")
            output_conf = gr.Textbox(label="Confidence Score")

    submit_btn.click(
        fn=classify_text,
        inputs=input_text,
        outputs=[output_label, output_conf],
    )

    gr.Markdown(
        "---\n"
        "⚠️ **Note:** The backend runs on Render's free tier, which spins down after inactivity. "
        "The first request may take 30-60 seconds while the server starts up.\n\n"
        "📊 **Model:** TF-IDF + HistGradientBoosting · 97.1% F1-Score · "
        "Trained on the [WELFake dataset](https://zenodo.org/record/4561253) (72K articles)\n\n"
        "🔗 [GitHub Repository](https://github.com/AyushKanyal-me/Fake-News-Detector)"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
