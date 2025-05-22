# Neural Search API with GPU Acceleration

A high-performance search API combining semantic understanding and traditional search techniques, powered by PyTorch and transformers.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/aminalaghband/local_search_api.git

# Navigate to project directory
cd local_search_api

# Build and launch containers
docker-compose up -d --build
Basic Usage
Generate an API key:

bash
curl -X POST http://localhost:8880/generate-key
Perform a search (replace <api_key> with your generated key):

bash
curl -X POST http://localhost:8880/search \
  -H "x-api-key: <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"q":"Dream"}'
Example Queries
bash
# Current time queries
curl ... -d '{"q":"current time in New York"}'

# Factual queries
curl ... -d '{"q":"population of Tokyo"}'

# Conceptual searches
curl ... -d '{"q":"philosophy of artificial intelligence"}'
Key Features
⚡ GPU-accelerated processing (NVIDIA CUDA supported)

Hybrid search combining semantic + keyword matching

Real-time web results with NLP enrichment

Automatic summarization & entity extraction

Troubleshooting
If you get no results:

Check Docker logs: docker-compose logs

Verify GPU access: nvidia-smi

Try simpler queries first

Configuration Options
Edit docker-compose.yml to:

Adjust GPU resource allocation

Change API port (default: 8880)

Modify search result limits


This version:
1. Starts with immediate executable commands
2. Uses your exact endpoint examples
3. Maintains clear formatting for copy-paste usage
4. Includes essential troubleshooting
5. Keeps the focus on practical usage rather than implementation details

Would you like me to add any specific configuration options or usage examples for your particular use case?