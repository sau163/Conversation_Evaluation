# Multi-stage Dockerfile for ML-Based Conversation Evaluation System
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .

# Copy pre-trained models and data
COPY data/processed/facets_structured.csv ./data/processed/
COPY data/processed/facet_index.pkl ./data/processed/
COPY data/models/evaluator.pkl ./data/models/
COPY data/processed/conversations.json ./data/processed/
COPY data/results/ ./data/results/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# API Image
FROM base as api
EXPOSE 8000
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
CMD ["python", "src/api/main.py"]

# UI Image  
FROM base as ui
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
