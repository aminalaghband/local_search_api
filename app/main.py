from fastapi import FastAPI, Header, HTTPException, status, Depends
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

# Configuration
DATABASE_URL = "sqlite:////app/data/local_search.db"
MEILI_HOST = os.getenv("MEILI_HOST", "http://meilisearch:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")
INDEX_NAME = "documents"
MIN_RESULTS_FOR_REFETCH = 3
MAX_DOCS_TO_INDEX = 50

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = spacy.blank("en")

# Database setup
database = databases.Database(DATABASE_URL)
metadata = sqlalchemy.MetaData()

# Table definitions
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

# Create engine & tables
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

def migrate_db():
    """Synchronous database migration for SQLAlchemy 1.4"""
    metadata.create_all(engine)

# FastAPI app
app = FastAPI(
    title="Intelligent Search API",
    version="2.0",
    description="An AI-enhanced search API with semantic understanding, personalization, and continuous learning"
)

# Pydantic models
class SearchRequest(BaseModel):
    q: str
    user_id: Optional[str] = None
    limit: Optional[int] = 10

class Feedback(BaseModel):
    query: str
    doc_id: str
    relevant: bool
    user_id: Optional[str] = None

class UserPreferences(BaseModel):
    user_id: str
    preferred_topics: List[str]
    preferred_sources: List[str]

# Startup & shutdown events
@app.on_event("startup")
async def startup():
    await database.connect()
    await verify_config()
    migrate_db()  # Synchronous table creation
    await initialize_meilisearch_index()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

async def verify_config():
    if not MEILI_MASTER_KEY:
        raise RuntimeError("MEILI_MASTER_KEY must be set in environment")
    if not MEILI_HOST:
        raise RuntimeError("MEILI_HOST environment variable not set")

async def initialize_meilisearch_index():
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {MEILI_MASTER_KEY}", "Content-Type": "application/json"}
        
        try:
            resp = await client.get(f"{MEILI_HOST}/indexes/{INDEX_NAME}", headers=headers)
            if resp.status_code == 404:
                await client.post(
                    f"{MEILI_HOST}/indexes",
                    headers=headers,
                    json={"uid": INDEX_NAME, "primaryKey": "id"}
                )
            
            # Configure index settings
            await client.post(
                f"{MEILI_HOST}/indexes/{INDEX_NAME}/settings",
                headers=headers,
                json={
                    "rankingRules": [
                        "words", "typo", "proximity", "attribute",
                        "sort", "exactness", "length:desc"
                    ],
                    "searchableAttributes": ["title", "content", "keywords", "entities"],
                    "sortableAttributes": ["length", "popularity"],
                    "filterableAttributes": ["language", "source_domain"],
                    "synonyms": {
                        "ai": ["artificial intelligence", "machine learning"],
                        "ml": ["machine learning"],
                        "it": ["information technology"]
                    }
                }
            )
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to initialize MeiliSearch: {str(e)}")

# Security helpers
def raise_invalid_key():
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key"
    )

async def verify_api_key(x_api_key: str = Header(...)):
    query = api_keys.select().where(api_keys.c.key == x_api_key)
    key = await database.fetch_one(query)
    if not key:
        raise_invalid_key()

# Intelligent document processing
def extract_entities(text: str) -> List[Dict[str, str]]:
    doc = nlp(text[:10000])  # Process first 10k chars for efficiency
    return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]

def extract_keywords(text: str) -> List[str]:
    doc = nlp(text[:10000])
    return list(set([chunk.text.lower() for chunk in doc.noun_chunks]))

def get_domain(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "")

def generate_documents_for_query(query: str) -> List[Dict[str, Any]]:
    search_url = "https://html.duckduckgo.com/html/"
    docs = []
    client = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}, timeout=15.0)
    
    try:
        resp = client.post(search_url, data={"q": query})
        soup = BeautifulSoup(resp.text, "html.parser")
        results = soup.find_all("div", class_="result", limit=10)

        for res in results:
            link_tag = res.find("a", class_="result__a")
            if not link_tag:
                continue
                
            link = link_tag.get("href")
            title = link_tag.get_text(strip=True)
            snippet = res.find("div", class_="result__snippet").get_text(strip=True) if res.find("div", class_="result__snippet") else ""

            # Enhanced content extraction
            content = ""
            try:
                downloaded = fetch_url(link)
                content = extract(downloaded, include_formatting=False, include_links=False) or snippet
                
                # Language detection
                lang = detect(content[:500]) if content else "en"
                
                # Entity and keyword extraction
                entities = extract_entities(content)
                keywords = extract_keywords(content)
                
                docs.append({
                    "id": abs(hash(link)) % (10**8),
                    "title": title,
                    "content": content,
                    "source": link,
                    "source_domain": get_domain(link),
                    "language": lang,
                    "entities": entities,
                    "keywords": keywords,
                    "length": len(content),
                    "popularity": len(keywords)  # Simple popularity metric
                })
                
                if len(docs) >= MAX_DOCS_TO_INDEX:
                    break
                    
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"Error fetching results: {str(e)}")
    finally:
        client.close()

    return docs or [{
        "id": abs(hash(query)) % (10**8),
        "title": f"No results for '{query}'",
        "content": "Try different search terms",
        "source": "",
        "source_domain": "",
        "language": "en",
        "entities": [],
        "keywords": [],
        "length": 0,
        "popularity": 0
    }]

# Query enhancement
async def enhance_query(query: str, user_id: Optional[str] = None) -> str:
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

# Personalization
async def personalize_results(user_id: Optional[str], results: dict) -> dict:
    if not user_id:
        return results
        
    prefs = await database.fetch_one(
        user_prefs.select().where(user_prefs.c.user_id == user_id)
    )
    
    if not prefs or not prefs["preferences"]:
        return results

    preferred_topics = prefs["preferences"].get("preferred_topics", [])
    preferred_sources = prefs["preferences"].get("preferred_sources", [])
    
    boosted_hits = []
    for hit in results["hits"]:
        # Boost score for preferred topics
        hit_keywords = set(kw.lower() for kw in hit.get("keywords", []))
        topic_matches = len(hit_keywords.intersection(set(t.lower() for t in preferred_topics)))
        
        # Boost score for preferred sources
        source_match = hit.get("source_domain", "") in preferred_sources
        
        if topic_matches or source_match:
            hit["_rankingScore"] *= (1 + (0.2 * topic_matches) + (0.3 if source_match else 0))
        boosted_hits.append(hit)
    
    results["hits"] = sorted(boosted_hits, key=lambda x: x["_rankingScore"], reverse=True)
    return results

# API Endpoints
@app.post("/generate-key")
async def generate_key():
    new_key = secrets.token_hex(16)
    await database.execute(api_keys.insert().values(key=new_key))
    return {"api_key": new_key}

@app.post("/search")
async def search(
    req: SearchRequest,
    x_api_key: str = Header(...)
):
    await verify_api_key(x_api_key)
    
    enhanced_q = await enhance_query(req.q, req.user_id)
    
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {MEILI_MASTER_KEY}"}
        
        try:
            # First try with MeiliSearch
            params = {
                "q": enhanced_q,
                "limit": req.limit,
                "attributesToRetrieve": ["*"],
                "attributesToHighlight": ["content"]
            }
            
            resp = await client.get(
                f"{MEILI_HOST}/indexes/{INDEX_NAME}/search",
                params=params,
                headers=headers
            )
            resp.raise_for_status()
            results = resp.json()
            
            # If insufficient results, fetch and index new content
            if len(results.get("hits", [])) < MIN_RESULTS_FOR_REFETCH:
                docs = generate_documents_for_query(req.q)
                if docs:
                    await client.post(
                        f"{MEILI_HOST}/indexes/{INDEX_NAME}/documents",
                        json=docs,
                        headers={**headers, "Content-Type": "application/json"}
                    )
                    # Search again with new content
                    resp = await client.get(
                        f"{MEILI_HOST}/indexes/{INDEX_NAME}/search",
                        params=params,
                        headers=headers
                    )
                    results = resp.json()
            
            # Personalize results if user_id provided
            if req.user_id:
                results = await personalize_results(req.user_id, results)
                
            # Record search history
            if req.user_id:
                await database.execute(
                    user_prefs.update()
                    .where(user_prefs.c.user_id == req.user_id)
                    .values(search_history=sqlalchemy.func.json_array_append(
                        user_prefs.c.search_history,
                        "$",
                        {"query": req.q, "timestamp": str(datetime.utcnow())}
                    ))
                )
            
            return results
            
        except httpx.HTTPStatusError as e:
            detail = "Search service error" if e.response.status_code >= 500 else e.response.text
            raise HTTPException(status_code=e.response.status_code, detail=detail)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )

@app.post("/feedback")
async def submit_feedback(
    fb: Feedback,
    x_api_key: str = Header(...)
):
    await verify_api_key(x_api_key)
    
    await database.execute(
        search_feedback.insert().values(
            query=fb.query,
            doc_id=fb.doc_id,
            relevant=fb.relevant,
            user_id=fb.user_id
        )
    )
    
    return {"status": "feedback recorded"}

@app.post("/user/preferences")
async def update_preferences(
    prefs: UserPreferences,
    x_api_key: str = Header(...)
):
    await verify_api_key(x_api_key)
    
    await database.execute(
        user_prefs.insert()
        .values(
            user_id=prefs.user_id,
            preferences={
                "preferred_topics": prefs.preferred_topics,
                "preferred_sources": prefs.preferred_sources
            }
        )
        .on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "preferences": {
                    "preferred_topics": prefs.preferred_topics,
                    "preferred_sources": prefs.preferred_sources
                }
            }
        )
    )
    
    return {"status": "preferences updated"}
