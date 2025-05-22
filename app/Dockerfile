FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA-compatible versions
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -m spacy download en_core_web_trf

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]