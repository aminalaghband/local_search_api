FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV CUDA_MEM_LIMIT=4096
# 1. Install system dependencies
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
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# 2. Install Python packages with pinned versions
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub==0.22.2 \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# 3. Download spaCy model
RUN python -m spacy download en_core_web_sm
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    huggingface-hub==0.22.2 \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# 3. Download models and create cache directories
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    mkdir -p /root/.cache/huggingface && \
    python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python3 -m spacy download en_core_web_sm && \
    python3 -m nltk.downloader punkt wordnet && \
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"

# 4. Copy application
COPY ./app /app

# 5. Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]