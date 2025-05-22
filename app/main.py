import torch
from fastapi import FastAPI, Header, HTTPException, status, Request
from pydantic import BaseModel
import httpx
import os
import databases
import sqlalchemy
import secrets
import spacy
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract
from langdetect import detect
from textblob import TextBlob
from nltk.corpus import wordnet
from thefuzz import fuzz
from datetime import datetime
import nltk
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from urllib.parse import urlparse
from fastapi.middleware.cors import CORSMiddleware

# Initialize CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing with {torch.cuda.get_device_name(0)}")

# Configuration
DATABASE_URL = "sqlite:////app/data/local_search.db"
MEILI_HOST = os.getenv("MEILI_HOST", "http://meilisearch:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")
INDEX_NAME = "documents"
MIN_RESULTS_FOR_REFETCH = 3
MAX_DOCS_TO_INDEX = 50

# Initialize models with mixed precision
semantic_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)
summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn", 
    device=0,
    torch_dtype=torch.float16
)
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    device=0,
    aggregation_strategy="average"
)

# Database setup
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

api_keys = sqlalchemy.Table(
    "api_keys",
    metadata,
    sqlalchemy.Column("key", sqlalchemy.String, primary_key=True),
)

user_prefs = sqlalchemy.Table(
    "user_prefs",
    metadata,
    sqlalchemy.Column("user_id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("preferences", sqlalchemy.JSON),
    sqlalchemy.Column("search_history", sqlalchemy.JSON),
)

search_feedback = sqlalchemy.Table(
    "search_feedback",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("query", sqlalchemy.String),
    sqlalchemy.Column("doc_id", sqlalchemy.String),
    sqlalchemy.Column("relevant", sqlalchemy.Boolean),
    sqlalchemy.Column("timestamp", sqlalchemy.DateTime, default=datetime.utcnow),
)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def migrate_db():
    metadata.create_all(engine)

app = FastAPI(
    title="Neural Search API",
    version="5.0",
    description="GPU-accelerated intelligent search engine"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    q: str
    user_id: Optional[str] = None
    limit: Optional[int] = 10
    neural: Optional[bool] = True

class EnhancedResult(BaseModel):
    id: int
    title: str
    content: str
    source: str
    summary: str
    entities: List[Dict]
    score: float
    processing_time_ms: float

class SearchResponse(BaseModel):
    results: List[EnhancedResult]
    query_analysis: Dict
    hardware: Dict

@app.on_event("startup")
async def startup():
    await database.connect()
    migrate_db()
    await initialize_meilisearch_index()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✅ Services initialized")

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    torch.cuda.empty_cache()

async def initialize_meilisearch_index():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {MEILI_MASTER_KEY}"}
        try:
            await client.post(
                f"{MEILI_HOST}/indexes",
                headers=headers,
                json={"uid": INDEX_NAME, "primaryKey": "id"}
            )
            await client.post(
                f"{MEILI_HOST}/indexes/{INDEX_NAME}/settings",
                headers=headers,
                json={
                    "filterableAttributes": ["language", "source_domain"],
                    "sortableAttributes": ["_geo"]
                }
            )
        except httpx.HTTPStatusError as e:
            print(f"⚠️ MeiliSearch initialization error: {e}")

def verify_api_key(key: str):
    query = api_keys.select().where(api_keys.c.key == key)
    return database.fetch_one(query) is not None

def gpu_extract_features(text: str) -> Dict:
    """Extract NLP features using GPU acceleration"""
    with torch.no_grad(), torch.cuda.amp.autocast():
        # Entity recognition
        entities = ner_pipeline(text[:512])
        
        # Summarization
        summary = summarizer(text[:1024], max_length=130)[0]["summary_text"]
        
        # Semantic embedding
        embedding = semantic_model.encode(text, convert_to_tensor=True)
        
        # Keywords via noun chunks
        doc = nlp(text[:10000])
        keywords = list(set([chunk.text for chunk in doc.noun_chunks]))
        
    return {
        "entities": entities,
        "summary": summary,
        "embedding": embedding,
        "keywords": keywords
    }

async def hybrid_search(query: str, documents: List[Dict]) -> List[Dict]:
    """Combine BM25 and semantic search scores"""
    # Lexical BM25
    tokenized_docs = [d["content"].split() for d in documents]
    bm25 = BM25Okapi(tokenized_docs)
    lexical_scores = bm25.get_scores(query.split())
    
    # Semantic similarity
    query_embed = semantic_model.encode(query, convert_to_tensor=True)
    doc_embeds = torch.stack([d["embedding"] for d in documents])
    semantic_scores = torch.nn.functional.cosine_similarity(
        query_embed, 
        doc_embeds
    ).cpu().numpy()
    
    # Combine scores
    combined = 0.6*semantic_scores + 0.4*lexical_scores
    ranked_indices = np.argsort(combined)[::-1]
    
    return [documents[i] for i in ranked_indices], combined[ranked_indices]

def generate_documents(query: str) -> List[Dict]:
    """Fetch and process search results"""
    client = httpx.Client(timeout=30.0)
    docs = []
    
    try:
        # DuckDuckGo scraping
        resp = client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"}
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for result in soup.find_all("div", class_="result", limit=MAX_DOCS_TO_INDEX):
            try:
                link = result.find("a", class_="result__a")["href"]
                title = result.find("a", class_="result__a").get_text(strip=True)
                snippet = result.find("div", class_="result__snippet").get_text() if result.find("div", class_="result__snippet") else ""
                
                # Fetch full content
                downloaded = fetch_url(link)
                content = extract(downloaded) or snippet
                
                # Process with GPU
                features = gpu_extract_features(content)
                
                docs.append({
                    "id": abs(hash(link)) % (10**8),
                    "title": title,
                    "content": content,
                    "source": link,
                    "source_domain": urlparse(link).netloc,
                    **features
                })
                
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"Search error: {str(e)}")
    finally:
        client.close()
    
    return docs or [create_empty_result(query)]

def create_empty_result(query: str) -> Dict:
    return {
        "id": abs(hash(query)) % (10**8),
        "title": f"No results for '{query}'",
        "content": "",
        "source": "",
        "source_domain": "",
        "entities": [],
        "summary": "",
        "keywords": [],
        "embedding": torch.zeros(384).to(DEVICE)
    }

@app.post("/search")
async def neural_search(
    request: SearchRequest, 
    x_api_key: str = Header(...)
):
    """Main search endpoint with GPU acceleration"""
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Start GPU timer
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    
    # Process query
    expanded_query = await expand_query(request.q)
    documents = generate_documents(expanded_query)
    
    # Hybrid ranking
    ranked_docs, scores = await hybrid_search(expanded_query, documents)
    
    # End GPU timer
    end_event.record()
    torch.cuda.synchronize()
    gpu_time = start_event.elapsed_time(end_event)
    
    # Build response
    return SearchResponse(
        results=[
            EnhancedResult(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                source=doc["source"],
                summary=doc["summary"],
                entities=doc["entities"],
                score=float(score),
                processing_time_ms=gpu_time
            )
            for doc, score in zip(ranked_docs[:request.limit], scores)
        ],
        query_analysis={
            "original": request.q,
            "expanded": expanded_query,
            "terms": expanded_query.split()
        },
        hardware={
            "device": torch.cuda.get_device_name(0),
            "memory_used": torch.cuda.memory_allocated(0),
            "compute_time_ms": gpu_time
        }
    )

@app.post("/generate-key")
async def generate_key():
    """Create new API key"""
    new_key = secrets.token_hex(32)
    await database.execute(api_keys.insert().values(key=new_key))
    return {"api_key": new_key}

@app.get("/system-status")
async def system_status():
    """Check GPU and service health"""
    return {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "memory": {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0)
            },
            "utilization": torch.cuda.utilization()
        },
        "services": {
            "database": "active",
            "search": "ready",
            "models": "loaded"
        }
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)