FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# 1. Install system dependencies (fixed line continuation)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    libgl1 \
    libglib2.0-0 \
    cuda-toolkit-12-1 \
    libcudnn8=8.9.2.26-1+cuda12.1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# 3. Download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python -m spacy download en_core_web_trf && \
    python -m nltk.downloader punkt wordnet

# 4. Copy application
COPY ./app /app

# 5. Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]