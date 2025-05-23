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
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import gc
import asyncio

# Initialize CUDA
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Initializing with {torch.cuda.get_device_name(0)}")

# Configuration
DATABASE_URL = "sqlite:////app/data/local_search.db"
MEILI_HOST = os.getenv("MEILI_HOST", "http://meilisearch:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")
INDEX_NAME = "documents"
MIN_RESULTS_FOR_REFETCH = 3
MAX_DOCS_TO_INDEX = 50
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "10.0"))  # Seconds

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

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj
@app.on_event("startup")
async def startup():
    await database.connect()
    migrate_db()
    await initialize_meilisearch_index()
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("Services initialized")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown"""
    await database.disconnect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Freed {torch.cuda.memory_allocated()//1024**2}MB GPU memory")
    
    print("Service shutdown complete")

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
    """Extract NLP features with proper CUDA tensor handling"""
    features = {
        "entities": [],
        "summary": "",
        "embedding": [],
        "keywords": []
    }
    
    try:
        with torch.no_grad(), torch.cuda.amp.autocast():
            # Entity recognition
            entities = ner_pipeline(text[:512])
            features["entities"] = [convert_numpy_types(e) for e in entities]
            
            # Summarization
            summary = summarizer(text[:1024], max_length=130)[0]["summary_text"]
            features["summary"] = str(summary)
            
            # Semantic embedding (convert to CPU numpy array)
            embedding = semantic_model.encode(text[:512], convert_to_tensor=True)
            features["embedding"] = embedding.cpu().numpy().tolist()
            
            # Keywords
            doc = nlp(text[:10000])
            features["keywords"] = list(set([str(chunk.text) for chunk in doc.noun_chunks]))
            
    except Exception as e:
        print(f"Feature extraction error: {str(e)}")
        features["embedding"] = []
    
    return features

def advanced_tokenize(query: str) -> List[str]:
    """Improved tokenization preserving phrases and entities"""
    # Preserve quoted phrases, numbers, and hyphenated words
    tokens = re.findall(r'\"(.+?)\"|(\d+-\w+)|(\w+-\d+)|(\b[A-Z][a-z]+\b)|(\d+)|(\w+)', query)
    
    # Flatten and filter matches
    cleaned = []
    for group in tokens:
        for match in group:
            if match:
                cleaned.append(match.strip('"'))
                break
    
    return cleaned

async def expand_query(query: str, user_id: Optional[str] = None) -> str:
    """Smarter query expansion preserving original structure"""
    try:
        # Preserve original casing for proper nouns
        original_terms = advanced_tokenize(query)
        doc = nlp(query)
        
        # Identify named entities and numbers
        preserved_terms = {
            ent.text.lower() for ent in doc.ents
        } | {
            token.text.lower() for token in doc if token.like_num
        }
        
        # Smart correction without losing context
        corrected = str(TextBlob(query).correct())
        if fuzz.ratio(query.lower(), corrected.lower()) > 89:
            query = corrected
    except:
        pass
    
    # Process terms with entity awareness
    final_terms = []
    for term in advanced_tokenize(query):
        term_lower = term.lower()
        
        # Preserve original if entity/number/phrase
        if (term_lower in preserved_terms or
            any(c.isupper() for c in term) or
            term.isdigit() or
            '-' in term):
            final_terms.append(term)
            continue
            
        # Add controlled synonyms
        syns = set()
        for syn in wordnet.synsets(term):
            for lemma in syn.lemmas():
                if lemma.name().lower() != term_lower:
                    syns.add(lemma.name().replace('_', ' '))
        
        # Add original + best synonym
        final_terms.append(term)
        if syns:
            final_terms.append(max(syns, key=lambda x: fuzz.ratio(term, x)))
    
    # Maintain original order with limited expansion
    return ' '.join(final_terms[:12])  # Max 12 terms

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
    """Wikipedia search with enhanced error handling"""
    docs = []
    try:
        search_results = wikipedia.search(query, results=2)
        for title in search_results[:2]:  # Limit to 2 results
            try:
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content[:10000]
                features = gpu_extract_features(content)
                docs.append({
                    "id": abs(hash(page.url)) % (10**8),
                    "title": str(page.title),
                    "content": str(content),
                    "source": str(page.url),
                    "source_domain": "wikipedia.org",
                    **features
                })
            except wikipedia.DisambiguationError as e:
                print(f"Wikipedia disambiguation error: {e.options[:3]}")
            except wikipedia.PageError:
                print(f"Wikipedia page not found: {title}")
            except Exception as e:
                print(f"Wikipedia processing error: {str(e)}")
    except Exception as e:
        print(f"Wikipedia search failed: {str(e)}")
    return docs

def generate_documents(query: str) -> List[Dict[str, Any]]:
    """Aggregate from multiple sources with error handling."""
    docs = []
    docs += generate_duckduckgo_documents(query)
    docs += generate_wikipedia_documents(query)
    # Add more sources here as needed
    return docs if docs else [create_empty_result(query)]

async def hybrid_search(query: str, documents: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Hybrid search with CUDA safety and timeout handling"""
    if not documents:
        return [], []
    
    valid_docs = [d for d in documents if d.get("content")]
    if not valid_docs:
        return [], []
    
    # Pre-process query embedding in main thread with CUDA
    try:
        # Get query embedding before moving to thread
        with torch.no_grad():
            query_embed = semantic_model.encode(query, convert_to_tensor=True).cpu().numpy()
    except Exception as e:
        print(f"Query embedding failed: {str(e)}")
        return [], []

    # Define CPU-only search logic
    def _cpu_search_core(query_embed: np.ndarray, valid_docs: List[Dict]):
        try:
            # Validate and convert document embeddings
            embeddings = []
            for doc in valid_docs:
                if isinstance(doc["embedding"], list):
                    emb = np.array(doc["embedding"], dtype=np.float32)
                elif torch.is_tensor(doc["embedding"]):
                    emb = doc["embedding"].cpu().numpy()
                else:
                    raise ValueError("Invalid embedding format")
                embeddings.append(emb)

            # BM25 lexical
            tokenized_docs = [doc["content"].split() for doc in valid_docs]
            bm25 = BM25Okapi(tokenized_docs)
            lexical_scores = np.array(bm25.get_scores(query.split()), dtype=np.float32)

            # Semantic similarity
            semantic_scores = cosine_similarity([query_embed], embeddings)[0].astype(np.float32)

            # Combine scores
            phrase_bonus = np.array([2.0 if query.lower() in doc["content"].lower() else 1.0 
                                   for doc in valid_docs], dtype=np.float32)
            domain_bonus = np.array([1.5 if doc.get("source_domain") == "wikipedia.org" else 1.0 
                                   for doc in valid_docs], dtype=np.float32)
            
            geo_boost = np.array([1.2 if any(ent['word'].lower() in ['us', 'usa', 'united states']
                     for ent in doc.get('entities', [])) else 1.0
              for doc in valid_docs], dtype=np.float32)
    
            combined = (0.5*semantic_scores + 0.5*lexical_scores) * phrase_bonus * domain_bonus * geo_boost
            return combined.tolist()
        
        except Exception as e:
            print(f"Search core error: {str(e)}")
            return None

    try:
        # Run CPU-intensive parts in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Pass pre-computed query embedding
            future = loop.run_in_executor(
                executor,
                _cpu_search_core,
                query_embed,
                valid_docs
            )
            combined = await asyncio.wait_for(future, timeout=10.0)

            if combined is None:
                return [], []

            ranked_indices = np.argsort(combined)[::-1]
            return (
                [valid_docs[i] for i in ranked_indices],
                [combined[i] for i in ranked_indices]
            )
            
    except TimeoutError:
        print("🕒 Search timed out after 10 seconds")
        return [], []
    except Exception as e:
        print(f"Hybrid search failed: {str(e)}")
        return [], []
    finally:
        # Cleanup resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def create_empty_result(query: str) -> Dict[str, Any]:
    return {
        "id": abs(hash(query)) % (10**8),
        "title": f"No results for '{query[:50]}'",
        "content": "Try: " + ", ".join([
            "Using more specific terms",
            "Checking spelling",
            "Adding location names",
            "Using fewer synonyms"
        ]),
        "source": "",
        "source_domain": "",
        "entities": [],
        "summary": "",
        "keywords": [],
        "embedding": [],
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
    """Main search endpoint with enhanced type safety"""
    if not await verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # GPU timing with conversion
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    
    try:
        expanded_query = await expand_query(request.q)
        documents = generate_documents(expanded_query)
        ranked_docs, scores = await hybrid_search(expanded_query, documents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search processing error: {str(e)}"
        )
    
    end_event.record()
    torch.cuda.synchronize()
    gpu_time = float(start_event.elapsed_time(end_event))

    # Convert all results to type-safe format
    safe_results = []
    for doc, score in zip(ranked_docs[:request.limit], scores):
        try:
            safe_results.append(EnhancedResult(
                id=int(doc["id"]),
                title=str(doc["title"]),
                content=str(doc["content"]),
                source=str(doc["source"]),
                summary=str(doc.get("summary", "")),
                entities=convert_numpy_types(doc.get("entities", [])),
                score=float(score),
                processing_time_ms=float(gpu_time)
            ))
        except Exception as e:
            print(f"Result conversion error: {str(e)}")
    
    return SearchResponse(
        results=safe_results,
        query_analysis={
            "original": str(request.q),
            "expanded": str(expanded_query),
            "terms": list(map(str, expanded_query.split()))
        },
        hardware={
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
            "memory_used": int(torch.cuda.memory_allocated(0)),
            "compute_time_ms": float(gpu_time)
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