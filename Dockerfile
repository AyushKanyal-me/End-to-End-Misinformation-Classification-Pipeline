# Use the official Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy the rest of the application code
COPY . .

# Expose default port
EXPOSE 8000

# Render sets $PORT dynamically; default to 8000 for local Docker usage
CMD uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}
