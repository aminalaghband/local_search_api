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

async def hybrid_search(query: str, documents: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Safe implementation of hybrid search with empty document handling"""
    if not documents:
        return [], []
    
    # Filter out empty documents
    valid_docs = [d for d in documents if d.get("content")]
    if not valid_docs:
        return [], []
    
    # Tokenize documents
    tokenized_docs = [doc["content"].split() for doc in valid_docs]
    tokenized_docs = [doc for doc in tokenized_docs if doc]  # Remove empty token lists
    
    # If no valid tokens, return empty results
    if not tokenized_docs:
        return [], []
    
    # Proceed with BM25 and semantic search
    try:
        bm25 = BM25Okapi(tokenized_docs)
        lexical_scores = bm25.get_scores(query.split())
        
        query_embed = semantic_model.encode(query, convert_to_tensor=True)
        doc_embeds = torch.stack([doc["embedding"] for doc in valid_docs])
        semantic_scores = torch.nn.functional.cosine_similarity(
            query_embed, 
            doc_embeds
        ).cpu().numpy()
        
        combined = 0.6*semantic_scores + 0.4*lexical_scores
        ranked_indices = np.argsort(combined)[::-1]
        
        return [valid_docs[i] for i in ranked_indices], combined[ranked_indices]
    except Exception as e:
        print(f"Search error: {str(e)}")
        return [], []

def generate_documents(query: str) -> List[Dict[str, Any]]:
    """Generate search documents with robust error handling and content validation"""
    search_url = "https://html.duckduckgo.com/html/"
    docs = []
    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30.0,
        follow_redirects=True
    )

    try:
        # Fetch search results
        resp = client.post(search_url, data={"q": query})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("div", class_="result", limit=10)

        for res in results:
            try:
                # Extract basic info
                link_tag = res.find("a", class_="result__a")
                if not link_tag:
                    continue
                
                link = link_tag.get("href", "")
                title = link_tag.get_text(strip=True)
                snippet = res.find("div", class_="result__snippet")
                snippet = snippet.get_text(strip=True) if snippet else ""

                # Fetch and process full content
                content = ""
                try:
                    downloaded = fetch_url(link, timeout=5.0)
                    if downloaded:
                        content = extract(
                            downloaded,
                            include_formatting=False,
                            include_links=False,
                            include_tables=False
                        ) or snippet
                except Exception as e:
                    content = snippet
                    print(f"Content extraction failed for {link}: {str(e)}")

                # Ensure minimum content requirements
                if not content.strip():
                    content = snippet if snippet else "No content available"

                # Process with GPU
                features = gpu_extract_features(content)
                
                docs.append({
                    "id": abs(hash(link)) % (10**8),
                    "title": title[:500],  # Limit title length
                    "content": content[:10000],  # Limit content length
                    "source": link,
                    "source_domain": get_domain(link),
                    **features
                })

                if len(docs) >= MAX_DOCS_TO_INDEX:
                    break

            except Exception as e:
                print(f"Error processing result: {str(e)}")
                continue

    except Exception as e:
        print(f"Search failed: {str(e)}")
    finally:
        client.close()

    # Return empty result if no documents found
    return docs if docs else [create_empty_result(query)]

def create_empty_result(query: str) -> Dict[str, Any]:
    """Create a placeholder result for empty searches"""
    return {
        "id": abs(hash(query)) % (10**8),
        "title": f"No results for '{query}'",
        "content": "Try different search terms",
        "source": "",
        "source_domain": "",
        "entities": [],
        "summary": "",
        "keywords": [],
        "embedding": torch.zeros(384).to(DEVICE),
        "score": 0.0
    }
    
def get_domain(url: str) -> str:
    """Extract domain from URL"""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return ""
        return parsed.netloc.replace("www.", "").split(":")[0]
    except:
        return ""
async def expand_query(query: str, user_id: Optional[str] = None) -> str:
    """Enhance search queries with spelling correction and expansion"""
    # Spelling correction
    try:
        corrected = str(TextBlob(query).correct())
        if fuzz.ratio(query, corrected) > 85:
            query = corrected
    except:
        pass

    # Query expansion using WordNet
    expanded = set(query.split())
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded.add(lemma.name().replace("_", " "))
    
    # Add user preferences if available
    if user_id:
        prefs = await database.fetch_one(
            user_prefs.select().where(user_prefs.c.user_id == user_id)
        )
        if prefs and prefs["preferences"]:
            expanded.update(prefs["preferences"].get("preferred_topics", []))

    return " ".join(list(expanded)[:10])  # Return top 10 terms
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