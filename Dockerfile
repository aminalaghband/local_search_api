FROM nvidia/cuda:12.1.1-base-ubuntu22.04

WORKDIR /app

# 1. Install system dependencies with proper Python symlink
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
    ln -s /usr/bin/python3 /usr/bin/python && \  # Add this critical line
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 2. Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip cache purge

# 3. Download models (using python3 explicitly)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" && \
    python3 -m spacy download en_core_web_trf && \
    python3 -m nltk.downloader punkt wordnet

# 4. Copy application
COPY ./app /app

# 5. Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]