from fastapi.testclient import TestClient
from app.api import app

# Create a TestClient using a context manager to trigger lifespan events
# Since models might not be trained during CI initially, we mock the ml_models dictionary.

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert response.json()["status"] == "healthy"

def test_predict_validation_error():
    with TestClient(app) as client:
        # Send text that is too short (min 10 chars required)
        response = client.post("/predict", json={"text": "short"})
        assert response.status_code == 422
