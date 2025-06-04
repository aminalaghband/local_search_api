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
MODEL_MAX_LENGTH = 512  # Adjust based on your VRAM
GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
}

# Initialize models with mixed precision
semantic_model = SentenceTransformer('all-MiniLM-L6-v2').to(DEVICE)

# Initialize Qwen for enhanced search understanding
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat",
        trust_remote_code=True
    )
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-0.5B-Chat",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print("✓ Qwen model loaded successfully")
except Exception as e:
    print(f"⚠️ Qwen model loading failed: {e}")
    qwen_model = None
    qwen_tokenizer = None

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
    """More precise tokenization preserving exact terms and phrases"""
    # Match: quoted phrases, technical terms, hyphenated terms, and exact words
    pattern = r'\"(.+?)\"|([A-Za-z]+\d+|\d+[A-Za-z]+)|(\w+-\w+)|([A-Z][a-z]+\s[A-Z][a-z]+|\b\w{2,}\b)'
    matches = re.finditer(pattern, query)
    return [match.group().strip('"') for match in matches if match.group()]

def analyze_query_type(query: str) -> Dict[str, Any]:
    """Analyze query to determine type and extract key information"""
    doc = nlp(query)
    
    # Initialize analysis
    analysis = {
        "is_question": False,
        "question_type": None,
        "key_entities": [],
        "target_concepts": set(),
        "temporal_context": "current"  # default to current
    }
    
    # Detect questions
    question_words = {"who", "what", "when", "where", "why", "how", "which"}
    first_word = doc[0].text.lower()
    if first_word in question_words or "?" in query:
        analysis["is_question"] = True
        analysis["question_type"] = first_word
    
    # Extract entities and concepts
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
            analysis["key_entities"].append({
                "text": ent.text,
                "type": ent.label_
            })
    
    # Detect temporal context
    time_indicators = {
        "latest": "current",
        "current": "current",
        "now": "current",
        "present": "current",
        "previous": "past",
        "former": "past",
        "last": "past",
        "recent": "current"
    }
    
    query_tokens = query.lower().split()
    for token in query_tokens:
        if token in time_indicators:
            analysis["temporal_context"] = time_indicators[token]
    
    # Extract key concepts
    analysis["target_concepts"].update([
        token.lemma_.lower() for token in doc
        if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 2
    ])
    
    return analysis

async def expand_query(query: str, user_id: Optional[str] = None) -> str:
    """Enhanced query expansion with question understanding"""
    try:
        # Analyze query type and structure
        query_analysis = analyze_query_type(query)
        doc = nlp(query)
        
        # Initialize terms with different priorities
        essential_terms = set()  # Must-have terms
        context_terms = set()    # Important context
        optional_terms = set()   # Nice-to-have expansions
        
        # Handle question-type queries
        if query_analysis["is_question"]:
            # For "who" questions about current roles
            if (query_analysis["question_type"] == "who" and 
                query_analysis["temporal_context"] == "current"):
                essential_terms.add("current")
                essential_terms.add("incumbent")
                if "president" in query.lower():
                    essential_terms.add("president")
                    context_terms.add("administration")
                    context_terms.add("elected")
        
        # Process entities and maintain case
        for token in doc:
            if (token.pos_ in ["PROPN"] or 
                token.ent_type_ in ["GPE", "LOC", "ORG", "PERSON"] or
                token.is_upper):
                essential_terms.add(token.text)  # Keep original case
            elif len(token.text) > 2:  # Meaningful common words
                context_terms.add(token.text.lower())
        
        # Add entity information
        for entity in query_analysis["key_entities"]:
            essential_terms.add(entity["text"])
        
        # Careful synonym expansion only for context terms
        for word in context_terms.copy():
            if word not in essential_terms:
                syns = wordnet.synsets(word)
                if syns:
                    # Add only highly relevant synonyms
                    syn = syns[0]
                    lemmas = [
                        lemma.name().replace('_', ' ') 
                        for lemma in syn.lemmas()
                        if lemma.name().lower() != word
                    ]
                    if lemmas and fuzz.ratio(word, lemmas[0]) > 85:
                        optional_terms.add(lemmas[0])
        
        # Build final query with term priorities
        final_terms = list(essential_terms)  # Essential terms first
        final_terms.extend(t for t in context_terms if t not in essential_terms)
        final_terms.extend(t for t in optional_terms if t not in essential_terms and t not in context_terms)
        
        # Limit total terms but ensure all essential terms are included
        max_terms = 10
        if len(final_terms) > max_terms:
            preserved_terms = list(essential_terms)
            remaining_slots = max_terms - len(preserved_terms)
            if remaining_slots > 0:
                preserved_terms.extend(final_terms[len(essential_terms):len(essential_terms) + remaining_slots])
            final_terms = preserved_terms
        
        return " ".join(final_terms)
        
    except Exception as e:
        print(f"Query expansion error: {str(e)}")
        return query  # Return original query if expansion fails


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
    """Enhanced DuckDuckGo scraping with factual query handling"""
    docs = []
    search_url = "https://html.duckduckgo.com/html/"
    
    # Analyze query type
    query_analysis = analyze_query_type(query)
    enhanced_query = query

    # Enhance query based on type
    if query_analysis["is_question"]:
        if query_analysis["question_type"] == "who":
            if "president" in query.lower() and query_analysis["temporal_context"] == "current":
                enhanced_query = f"current incumbent {query}"
        elif query_analysis["question_type"] == "when":
            enhanced_query = f"date time {query}"
            
    # Additional context for temporal queries
    if query_analysis["temporal_context"] == "current":
        enhanced_query += " 2025 current incumbent"
        
    client = httpx.Client(
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9"
        },
        timeout=30.0,
        follow_redirects=True
    )
    
    try:
        # Add search parameters
        resp = client.post(search_url, data={
            "q": enhanced_query,
            "kl": "us-en",
            "k1": "-1",  # Disable safe search
            "kz": "1",   # Show more results
            "kaf": "1"   # Full content
        })
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Score and filter results
        scored_results = []
        seen_domains = set()
        
        # Try multiple result containers
        results = []
        for class_name in ["result", "web-result", "links_main", "result__body"]:
            results.extend(soup.find_all("div", class_=class_name))
            
        for res in results:
            try:
                link_tag = (res.find("a", class_="result__a") or 
                          res.find("a", class_="web-result__title") or
                          res.find("a", href=True))
                if not link_tag:
                    continue
                    
                link = link_tag.get("href", "")
                if not link or link.startswith(('javascript:', 'data:', '#')):
                    continue
                    
                domain = get_domain(link)
                if domain in seen_domains or not domain:
                    continue
                
                title = link_tag.get_text(strip=True)
                
                # Extract content
                snippet = None
                for snippet_class in ["result__snippet", "web-result__snippet", "result__body"]:
                    snippet_tag = res.find("div", class_=snippet_class)
                    if snippet_tag:
                        snippet = snippet_tag.get_text(strip=True)
                        break
                        
                content = snippet or ""
                
                # For factual queries, prefer recent and authoritative sources
                relevance_score = 0
                
                # Boost authoritative domains
                authority_domains = {
                    'wikipedia.org': 3,
                    'whitehouse.gov': 5,
                    'senate.gov': 4,
                    'house.gov': 4,
                    'state.gov': 4,
                    'ap.org': 3,
                    'reuters.com': 3,
                    'bbc.com': 3,
                    'nytimes.com': 3
                }
                
                for auth_domain, score in authority_domains.items():
                    if auth_domain in domain:
                        relevance_score += score
                        break
                
                # Boost results with recent dates for current queries
                if query_analysis["temporal_context"] == "current":
                    current_year = "2025"
                    if current_year in content[:200]:  # Check start of content
                        relevance_score += 2
                        
                # Try to fetch full content for promising results
                if (len(content) < 100 and relevance_score > 0) or len(content) < 50:
                    try:
                        downloaded = fetch_url(link, timeout=5.0)
                        if downloaded:
                            extracted = extract(downloaded)
                            if extracted and len(extracted) > len(content):
                                content = extracted
                    except Exception as e:
                        print(f"Content fetch error for {link}: {e}")
                
                if len(content) < 50:  # Skip if no meaningful content
                    continue
                    
                # Extract features
                features = gpu_extract_features(content)
                
                scored_results.append((relevance_score, {
                    "id": abs(hash(link)) % (10**8),
                    "title": title[:500],
                    "content": content[:10000],
                    "source": link,
                    "source_domain": domain,
                    "query_match": {
                        "is_question": query_analysis["is_question"],
                        "question_type": query_analysis["question_type"],
                        "temporal_context": query_analysis["temporal_context"]
                    },
                    **features
                }))
                
                seen_domains.add(domain)
                
            except Exception as e:
                print(f"Result processing error: {str(e)}")
                continue
                
        # Sort by relevance score and return
        scored_results.sort(reverse=True)
        docs = [doc for score, doc in scored_results]
                
    except Exception as e:
        print(f"DuckDuckGo search failed: {str(e)}")
        
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

async def generate_documents_async(query: str) -> List[Dict[str, Any]]:
    """Aggregate from multiple sources with strong prioritization of web results"""
    loop = asyncio.get_event_loop()
    
    # First try DuckDuckGo with more thorough scraping
    ddg_docs = await loop.run_in_executor(None, generate_duckduckgo_documents, query)
    
    # Filter out low-quality results and duplicates
    filtered_docs = []
    seen_domains = set()
    
    for doc in ddg_docs:
        domain = get_domain(doc["source"])
        
        # Skip if no meaningful content or from same domain
        if (len(doc.get("content", "")) < 50 or  # Too short
            domain in seen_domains or  # Duplicate domain
            not domain or  # Invalid domain
            "wikipedia.org" in domain):  # Skip Wikipedia from DuckDuckGo
            continue
            
        filtered_docs.append(doc)
        seen_domains.add(domain)
        
        if len(filtered_docs) >= 8: # Good number of diverse results
            return filtered_docs
    
    # If we don't have enough good results, try Wikipedia as backup
    if len(filtered_docs) < 3:
        wiki_docs = await loop.run_in_executor(None, generate_wikipedia_documents, query)
        filtered_docs.extend(wiki_docs[:1])  # Only add 1 Wikipedia result max
    
    return filtered_docs if filtered_docs else [create_empty_result(query)]

async def hybrid_search(query: str, documents: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Generic hybrid search combining semantic and lexical matching"""
    if not documents:
        return [], []
    
    valid_docs = [d for d in documents if d.get("content")]
    if not valid_docs:
        return [], []
    
    query_terms = query.lower().split()
    
    try:
        # Query embedding
        with torch.no_grad():
            query_embed = semantic_model.encode(
                query, 
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu().numpy()
            
    except Exception as e:
        print(f"Query embedding failed: {str(e)}")
        return [], []

    def _cpu_search_core(query_embed: np.ndarray, valid_docs: List[Dict]):
        try:
            # Process document embeddings
            embeddings = []
            for doc in valid_docs:
                try:
                    if isinstance(doc["embedding"], list):
                        emb = np.array(doc["embedding"], dtype=np.float32)
                    elif torch.is_tensor(doc["embedding"]):
                        emb = doc["embedding"].cpu().numpy()
                    else:
                        emb = np.zeros(semantic_model.get_sentence_embedding_dimension(), dtype=np.float32)
                    embeddings.append(emb)
                except:
                    embeddings.append(np.zeros_like(query_embed))

            # BM25 scoring
            tokenized_docs = [doc["content"].split() for doc in valid_docs]
            bm25 = BM25Okapi(tokenized_docs)
            lexical_scores = np.array(bm25.get_scores(query_terms), dtype=np.float32)

            # Semantic scoring
            try:
                semantic_scores = cosine_similarity([query_embed], embeddings)[0].astype(np.float32)
            except:
                semantic_scores = np.zeros(len(embeddings), dtype=np.float32)

            # Balanced scoring with title relevance
            title_relevance = np.array([
                1.5 if any(term in doc.get("title", "").lower() for term in query_terms) else 1.0
                for doc in valid_docs
            ], dtype=np.float32)

            # Combine scores with balanced weights
            combined = (
                (0.5 * semantic_scores) + 
                (0.5 * lexical_scores)
            ) * title_relevance

            return combined.tolist()
        
        except Exception as e:
            print(f"Search core error: {str(e)}")
            return None

    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                _cpu_search_core,
                query_embed,
                valid_docs
            )
            combined = await asyncio.wait_for(future, timeout=15.0)

            if not combined:
                return [], []

            ranked_indices = np.argsort(combined)[::-1]
            return (
                [valid_docs[i] for i in ranked_indices],
                [combined[i] for i in ranked_indices]
            )
            
    except TimeoutError:
        print("🕒 Search timed out after 15 seconds")
        return [], []
    except Exception as e:
        print(f"Hybrid search failed: {str(e)}")
        return [], []
    finally:
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
    """Main search endpoint with async document fetching and LLM enhancement"""
    if not await verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # GPU timing with conversion
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()

    try:
        expanded_query = await expand_query(request.q, request.user_id)
        documents = await generate_documents_async(expanded_query)
        ranked_docs, scores = await hybrid_search(expanded_query, documents)
        
        # Add scores to documents
        for doc, score in zip(ranked_docs, scores):
            doc["score"] = float(score)
            
        # Enhance results with LLM if available
        if qwen_model and len(ranked_docs) > 0:
            ranked_docs = enhance_search_with_llm(request.q, ranked_docs)
            scores = [doc.get("score", 0.0) for doc in ranked_docs]
            
    except Exception as e:
        print(f"Search processing error: {str(e)}")
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

    # Free GPU memory after search
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return SearchResponse(
        results=safe_results,
        query_analysis={
            "original": str(request.q),
            "expanded": str(expanded_query),
            "terms": list(map(str, expanded_query.split())),
            "llm_enhanced": qwen_model is not None
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

def enhance_search_with_llm(query: str, top_results: List[Dict]) -> List[Dict]:
    """Enhanced LLM-based result evaluation with factual query handling"""
    if not qwen_model or not qwen_tokenizer or not top_results:
        return top_results
        
    try:
        # Analyze query type
        query_analysis = analyze_query_type(query)
        
        # Create a context-aware prompt
        if query_analysis["is_question"]:
            if query_analysis["temporal_context"] == "current":
                prompt = f"""As a search evaluator in 2025, analyze these results for the question: "{query}"
                Focus on:
                1. Current and up-to-date information (as of 2025)
                2. Direct answers to the question
                3. Source authority and reliability
                4. Factual accuracy
                
                For each result, determine:
                1. Does it directly answer the question? (0-10)
                2. Is the information current? (0-10)
                3. Is the source authoritative? (0-10)
                
                Evaluate each result and assign a combined score (0-10).
                """
            else:
                prompt = f"""As a search evaluator, analyze these results for the question: "{query}"
                Focus on:
                1. Direct answers to the question
                2. Source authority and reliability
                3. Factual accuracy and detail
                4. Historical context when relevant
                
                For each result, determine:
                1. Does it directly answer the question? (0-10)
                2. Is the source authoritative? (0-10)
                3. Is the information complete? (0-10)
                
                Evaluate each result and assign a combined score (0-10).
                """
        else:
            prompt = f"""As a search evaluator, analyze these results for the query: "{query}"
            Focus on:
            1. Relevance to the query
            2. Information quality and completeness
            3. Source reliability
            
            Rate each result's relevance from 0-10.
            """
        
        # Add result summaries to prompt
        for i, doc in enumerate(top_results[:5]):
            prompt += f"\n\nResult {i+1}:\n"
            prompt += f"Title: {doc['title']}\n"
            prompt += f"Source: {doc['source_domain']}\n"
            prompt += f"Summary: {doc['summary'][:200]}...\n"
        
        # Generate evaluation
        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = qwen_tokenizer(prompt, return_tensors="pt").to(DEVICE)
            output = qwen_model.generate(
                **inputs,
                max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                temperature=GENERATION_CONFIG["temperature"],
                top_p=GENERATION_CONFIG["top_p"],
                do_sample=GENERATION_CONFIG["do_sample"]
            )
            evaluation = qwen_tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract scores and boost factual answers
        scores = []
        for line in evaluation.split("\n"):
            if "score:" in line.lower() or "combined:" in line.lower():
                try:
                    score = float(re.search(r"(\d+\.?\d*)", line).group(1))
                    scores.append(min(10.0, max(0.0, score)))
                except:
                    scores.append(5.0)
        
        # Adjust result ranking
        if scores and len(scores) == len(top_results[:5]):
            # Normalize scores
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            # Apply scores with factual query boost
            for i, score in enumerate(scores):
                normalized_score = (score - min_score) / score_range
                original_score = top_results[i].get("score", 0)
                
                # Higher LLM weight for factual queries
                if query_analysis["is_question"]:
                    llm_weight = 0.4  # 40% LLM, 60% original for factual queries
                else:
                    llm_weight = 0.3  # 30% LLM, 70% original for other queries
                    
                top_results[i]["score"] = (
                    (1 - llm_weight) * original_score + 
                    llm_weight * normalized_score
                )
            
            # Re-sort results
            top_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
    except Exception as e:
        print(f"LLM enhancement error: {str(e)}")
        
    return top_results