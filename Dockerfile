FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# System dependencies with cleanup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    libcudnn8 \
    cuda-toolkit-12-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies with pinned versions
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Pre-download models
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', device='cuda'); \
import spacy; spacy.cli.download('en_core_web_trf'); \
import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# Application files
COPY app/main.py .

# Health check and optimized run command
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/system-status || exit 1
# Optimal workers for GPU: (CPU cores * 2) + 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "60"]