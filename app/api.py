import os
import sys
import joblib
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import custom transformers so joblib can unpickle
from src.train import DenseTransformer
from src.train_hf import HuggingFaceEmbedder

# Inject into __main__ to avoid unpickling errors during pytest or uvicorn
setattr(sys.modules['__main__'], 'DenseTransformer', DenseTransformer)
setattr(sys.modules['__main__'], 'HuggingFaceEmbedder', HuggingFaceEmbedder)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The news article text to verify.")

class PredictResponse(BaseModel):
    prediction: str
    confidence: float

# Global model state
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the ML model(s) before the server starts accepting requests.
    Prioritizes the Hugging Face model if available, falls back to classical.
    """
    hf_model_path = os.path.join(PROJECT_ROOT, "models", "hf_pipeline.joblib")
    classical_model_path = os.path.join(PROJECT_ROOT, "models", "best_pipeline.joblib")
    
    if os.path.exists(hf_model_path):
        logger.info(f"Loading Hugging Face model from {hf_model_path}...")
        ml_models["pipeline"] = joblib.load(hf_model_path)
    elif os.path.exists(classical_model_path):
        logger.info(f"Loading Classical model from {classical_model_path}...")
        ml_models["pipeline"] = joblib.load(classical_model_path)
    else:
        logger.warning("No trained models found. The API will start, but predictions will fail.")
        
    yield
    
    # Cleanup on shutdown
    ml_models.clear()

app = FastAPI(
    title="Misinformation Classification API",
    description="REST API for classifying news articles as Real or Fake using TF-IDF + HistGradientBoosting. Trained on the WELFake dataset (72K articles).",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the Swagger API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API and model are healthy."""
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in ml_models
    }

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Classify a news article as Real or Fake.
    """
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train a model first.")
        
    pipeline = ml_models["pipeline"]
    
    # Predict
    try:
        # WELFake dataset: label 0 = Real, label 1 = Fake
        label = "Fake" if prediction == 1 else "Real"
        
        # Try to get confidence score
        try:
            proba = pipeline.predict_proba([request.text])[0]
            confidence = max(proba) * 100
        except AttributeError:
            confidence = 100.0  # Fallback if probability is unsupported
            
        return PredictResponse(prediction=label, confidence=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
