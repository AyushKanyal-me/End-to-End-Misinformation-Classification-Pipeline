# 🚀 End-to-End Misinformation Classification Pipeline

An end-to-end Machine Learning pipeline that classifies news articles as **Real** or **Fake**. Trained on the **WELFake dataset (72,134 articles)**, this project demonstrates production-grade ML Engineering practices, including modular code, experiment tracking, a REST API, and containerization.

## 🚀 Features

- **Classical & Deep Learning:** Compares 5 classical ML models (TF-IDF + Scikit-Learn) with modern Hugging Face semantic embeddings (`SentenceTransformers`).
- **MLOps & Tracking:** Integrated with **MLflow** to track metrics, hyperparameters, and model artifacts.
- **REST API:** Production-ready **FastAPI** backend with async model loading, Pydantic validation, and CORS.
- **Containerized:** Includes a multi-stage **Dockerfile** for easy deployment.
- **Automated Testing:** Unit tests for APIs and preprocessing pipelines.

## 📈 Performance & Impact

- **High Accuracy:** Achieved **97.1% Accuracy** and **97.1% F1-Score** on the holdout test set using TF-IDF vectorized features and a `HistGradientBoosting` classifier.
- **Low Latency:** Engineered for production with an asynchronous **FastAPI** backend, achieving **<40ms inference latency** per article.
- **Scalable Architecture:** Fully containerized inference API ready for cloud deployment (AWS/GCP) via Docker.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Frameworks** | scikit-learn, SentenceTransformers (Hugging Face) |
| **MLOps** | MLflow |
| **API Backend** | FastAPI, Uvicorn, Pydantic |
| **Deployment** | Docker |
| **Data & NLP** | Pandas, NumPy, NLTK |

## 🏗️ Project Architecture

```
Raw Text → TextPreprocessor → TF-IDF / HuggingFace Embedder → Classifier → FastAPI Inference
```

## 📸 System Visuals

### 1. Model Evaluation (Confusion Matrix & ROC)
| Confusion Matrix | ROC Curve |
| :---: | :---: |
| <img src="reports/confusion_matrix.png" width="400"/> | <img src="reports/roc_curve.png" width="400"/> |

### 2. FastAPI Inference Endpoint
*(Note: Replace this placeholder with a screenshot of your `http://localhost:8000/docs` Swagger UI)*
![FastAPI Swagger UI Placeholder](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)


## 💻 How to Run Locally

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
Because model files are large, they are not stored in this repository. You must generate them first:
```bash
# To train Classical ML models (Logistic Regression, Random Forest, etc.)
python main.py --step all_classical

# OR to train the Hugging Face model
python main.py --step train_hf
```

### 3. Start the API Server
```bash
python main.py --step api
```
Visit `http://localhost:8000/docs` to interact with the API via the Swagger UI.

## 🐳 Docker Deployment

To run this project as a containerized microservice:

1. Ensure you have trained a model (Step 2 above).
2. Build and run the image:
```bash
docker build -t fake-news-detector .
docker run -p 8000:8000 fake-news-detector
```

## 📊 Models Evaluated

- **Logistic Regression** (Baseline)
- **Multinomial Naive Bayes**
- **Linear SVM**
- **Random Forest**
- **HistGradientBoosting**
- **Hugging Face (`all-MiniLM-L6-v2`) + GBM**

> **Note:** The `data/` and `models/` directories are intentionally ignored in version control due to size constraints. Running the pipeline via `main.py` will automatically download the dataset (WELFake) and generate the models.
