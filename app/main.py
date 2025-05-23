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
from typing import Tuple
from typing import Any, Optional
import spacy
import re
import wikipedia

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
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model isn't downloaded, download it first
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
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

async def verify_api_key(key: str):
    query = api_keys.select().where(api_keys.c.key == key)
    result = await database.fetch_one(query)
    return result is not None

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

def advanced_tokenize(query: str) -> List[str]:
    """Tokenize query preserving quoted phrases and numbers."""
    # Preserve phrases in quotes and numbers as single tokens
    tokens = re.findall(r'"[^"]+"|\d+|\w+', query)
    return [t.strip('"') for t in tokens]

def weighted_expand_terms(tokens: List[str]) -> List[str]:
    """Expand terms with weights: phrases > numbers > words."""
    expanded = []
    for token in tokens:
        if token.isdigit():
            expanded.append(token)
            continue
        if len(token.split()) > 1:  # phrase
            expanded.append(token)
            continue
        if is_named_entity(token) or len(token) < 4:
            expanded.append(token)
            continue
        syns = wordnet.synsets(token)
        lemmas = set()
        for syn in syns:
            for lemma in syn.lemmas():
                if lemma.name().lower() != token.lower():
                    lemmas.add(lemma.name().replace("_", " "))
        # Add original and up to 1 synonym
        expanded.append(token)
        if lemmas:
            expanded.append(list(lemmas)[0])
    return expanded[:10]

async def expand_query(query: str, user_id: Optional[str] = None) -> str:
    """Advanced query expansion with phrase and number handling."""
    try:
        corrected = str(TextBlob(query).correct())
        if fuzz.ratio(query, corrected) > 85:
            query = corrected
    except:
        pass
    tokens = advanced_tokenize(query)
    expanded = set(weighted_expand_terms(tokens))
    # Add user preferences if available
    if user_id:
        prefs = await database.fetch_one(
            user_prefs.select().where(user_prefs.c.user_id == user_id)
        )
        if prefs and prefs["preferences"]:
            expanded.update(prefs["preferences"].get("preferred_topics", []))
    return " ".join(list(expanded)[:10])

def generate_duckduckgo_documents(query: str) -> List[Dict[str, Any]]:
    """DuckDuckGo scraping with error handling."""
    docs = []
    search_url = "https://html.duckduckgo.com/html/"
    client = httpx.Client(
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30.0,
        follow_redirects=True
    )
    try:
        resp = client.post(search_url, data={"q": query})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("div", class_="result", limit=10)
        if not results:
            results = soup.find_all("div", class_="web-result")
        for res in results:
            try:
                link_tag = res.find("a", class_="result__a") or res.find("a", class_="web-result__title")
                if not link_tag:
                    continue
                link = link_tag.get("href", "")
                title = link_tag.get_text(strip=True)
                snippet = res.find("div", class_="result__snippet") or res.find("div", class_="web-result__snippet")
                snippet = snippet.get_text(strip=True) if snippet else ""
                content = snippet
                try:
                    downloaded = fetch_url(link, timeout=5.0)
                    if downloaded:
                        content = extract(downloaded) or snippet
                except Exception:
                    pass
                features = gpu_extract_features(content)
                docs.append({
                    "id": abs(hash(link)) % (10**8),
                    "title": title[:500],
                    "content": content[:10000],
                    "source": link,
                    "source_domain": get_domain(link),
                    **features
                })
            except Exception as e:
                print(f"DuckDuckGo result error: {e}")
                continue
    except Exception as e:
        print(f"DuckDuckGo search failed: {e}")
    finally:
        client.close()
    return docs

def generate_wikipedia_documents(query: str) -> List[Dict[str, Any]]:
    """Wikipedia fallback source."""
    docs = []
    try:
        search_results = wikipedia.search(query, results=2)
        for title in search_results:
            try:
                page = wikipedia.page(title)
                content = page.content[:10000]
                features = gpu_extract_features(content)
                docs.append({
                    "id": abs(hash(page.url)) % (10**8),
                    "title": page.title,
                    "content": content,
                    "source": page.url,
                    "source_domain": "wikipedia.org",
                    **features
                })
            except Exception as e:
                print(f"Wikipedia fetch failed: {e}")
    except Exception as e:
        print(f"Wikipedia search failed: {e}")
    return docs

def generate_documents(query: str) -> List[Dict[str, Any]]:
    """Aggregate from multiple sources with error handling."""
    docs = []
    docs += generate_duckduckgo_documents(query)
    docs += generate_wikipedia_documents(query)
    # Add more sources here as needed
    return docs if docs else [create_empty_result(query)]

async def hybrid_search(query: str, documents: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Hybrid scoring with phrase and domain weighting."""
    if not documents:
        return [], []
    valid_docs = [d for d in documents if d.get("content")]
    if not valid_docs:
        return [], []
    # Phrase bonus: boost docs containing exact query phrase
    phrase_bonus = np.array([2.0 if query.lower() in doc["content"].lower() else 1.0 for doc in valid_docs])
    # Domain weighting: boost Wikipedia
    domain_bonus = np.array([1.5 if doc.get("source_domain") == "wikipedia.org" else 1.0 for doc in valid_docs])
    # BM25 lexical
    tokenized_docs = [doc["content"].split() for doc in valid_docs]
    bm25 = BM25Okapi(tokenized_docs)
    lexical_scores = np.array(bm25.get_scores(query.split()))
    # Semantic
    query_embed = semantic_model.encode(query, convert_to_tensor=True)
    doc_embeds = torch.stack([doc["embedding"] for doc in valid_docs])
    semantic_scores = torch.nn.functional.cosine_similarity(query_embed, doc_embeds).cpu().numpy()
    # Combine
    combined = (0.5 * semantic_scores + 0.5 * lexical_scores) * phrase_bonus * domain_bonus
    ranked_indices = np.argsort(combined)[::-1]
    return [valid_docs[i] for i in ranked_indices], combined[ranked_indices]

def create_empty_result(query: str) -> Dict[str, Any]:
    """Better empty result handling."""
    return {
        "id": abs(hash(query)) % (10**8),
        "title": f"No results for '{query}'",
        "content": "Try different search terms or check your spelling.",
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
def is_named_entity(word: str) -> bool:
    doc = nlp(word)
    return any(ent.label_ in ["GPE", "ORG", "PERSON", "LOC"] for ent in doc.ents)

@app.post("/search")
async def neural_search(
    request: SearchRequest, 
    x_api_key: str = Header(...)
):
    """Main search endpoint with GPU acceleration"""
    if not await verify_api_key(x_api_key):
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
    gpu_time = float(start_event.elapsed_time(end_event))  # Ensure this is a Python float

    # Robust conversion for all numpy/tensor types
    def to_pyfloat(val):
        try:
            if hasattr(val, "item"):
                return float(val.item())
            if isinstance(val, (np.floating, np.integer)):
                return float(val)
            return float(val)
        except Exception:
            return 0.0

    py_scores = [to_pyfloat(s) for s in scores]

    return SearchResponse(
        results=[
            EnhancedResult(
                id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                source=doc["source"],
                summary=doc["summary"],
                entities=doc["entities"],
                score=to_pyfloat(score),
                processing_time_ms=to_pyfloat(gpu_time)
            )
            for doc, score in zip(ranked_docs[:request.limit], py_scores)
        ],
        query_analysis={
            "original": request.q,
            "expanded": expanded_query,
            "terms": expanded_query.split()
        },
        hardware={
            "device": torch.cuda.get_device_name(0),
            "memory_used": int(torch.cuda.memory_allocated(0)),
            "compute_time_ms": to_pyfloat(gpu_time)
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