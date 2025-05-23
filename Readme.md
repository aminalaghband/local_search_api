# Neural Search API with GPU Acceleration

A high-performance, production-ready search API that combines semantic understanding and traditional keyword search, powered by PyTorch, transformers, and GPU acceleration. Designed for fast, accurate, and scalable information retrieval across diverse domains.

---

## 🚀 Features

- **⚡ GPU-Accelerated Processing:** Leverages NVIDIA CUDA for high-throughput inference.
- **🔍 Hybrid Search:** Combines semantic (vector-based) and keyword (BM25/traditional) matching for superior relevance.
- **🌐 Real-Time Web Results:** Optionally enriches queries with up-to-date web data and NLP processing.
- **📝 Summarization & Entity Extraction:** Automatic summarization and entity recognition for retrieved documents.
- **🔒 API Key Security:** Secure endpoints with API key authentication.
- **📦 Dockerized:** Easy deployment with Docker and Docker Compose.
- **🛠️ Configurable:** Tune search limits, GPU allocation, and ports via `docker-compose.yaml`.

---

## 📦 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/aminalaghband/local_search_api.git
cd local_search_api
```

### 2. Build & Launch with Docker

```bash
docker-compose up -d --build
```

### 3. Generate an API Key

```bash
curl -X POST http://localhost:8880/generate-key
```

### 4. Perform a Search

Replace `<api_key>` with your generated key:

```bash
curl -X POST http://localhost:8880/search \
  -H "x-api-key: <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"q":"Dream"}'
```

---

## 🧑‍💻 Example Queries

```bash
# Current time queries
curl -X POST ... -d '{"q":"current time in New York"}'

# Factual queries
curl -X POST ... -d '{"q":"population of Tokyo"}'

# Conceptual searches
curl -X POST ... -d '{"q":"philosophy of artificial intelligence"}'
```

---

## 🛠️ Configuration

Edit `docker-compose.yaml` to:

- Adjust GPU resource allocation
- Change API port (default: `8880`)
- Modify search result limits

You can also edit environment variables or config files in `app/` for advanced settings.

---

## 🏗️ Architecture

- **API Layer:** FastAPI-based, handles authentication, query parsing, and response formatting.
- **Semantic Search:** Uses transformer-based embeddings for vector similarity.
- **Keyword Search:** BM25 or similar algorithm for traditional text matching.
- **Hybrid Ranking:** Combines both scores for optimal relevance.
- **Optional Web/NLP Enrichment:** Fetches and processes real-time web data.
- **Database:** Local SQLite (`local_search.db`) for fast document retrieval.

---

## 🔒 Security

- All search endpoints require a valid API key via the `x-api-key` header.
- Keys are generated via the `/generate-key` endpoint.

---

## 📝 API Reference

### `POST /generate-key`
- **Description:** Generates a new API key.
- **Response:** `{ "api_key": "<your_key>" }`

### `POST /search`
- **Headers:** `x-api-key: <your_key>`, `Content-Type: application/json`
- **Body:** `{ "q": "<your query>" }`
- **Response:** JSON array of ranked search results with optional summaries/entities.

---

## 🩺 Troubleshooting

- **No results?**
  - Check logs: `docker-compose logs`
  - Verify GPU access: `nvidia-smi`
  - Try simpler queries

- **API not responding?**
  - Ensure containers are running: `docker ps`
  - Check port conflicts in `docker-compose.yaml`

---

## 🧩 Extending

- Add new data to `data/local_search.db` or `app/data/local_search.db`.
- Customize models or ranking logic in `app/main.py`.

---

## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---

## ✨ Acknowledgements

- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)