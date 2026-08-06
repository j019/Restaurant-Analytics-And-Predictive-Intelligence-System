"""
Zomato RAG Chatbot — Refactored CSV-driven Pipeline (Memory-Optimized for Cloud Deployment)
===================================================
Refactored to provide natural, humane, and conversational responses:
  - Custom System Prompt: Enforces warm, assistant-style synthesis rather
    than metadata dumping or quoting raw token arrays.
  - Temperature Tuning: Set to 0.4 for a natural conversational balance.
  - Narrative Document Builder: Constructs human-readable narrative sentences
    prior to vector embedding.
  - Cloud Optimization: Selective column loading & row sampling for 512MB RAM environments.
"""

import sys
import math
import os
import urllib.request
from pathlib import Path
from ast import literal_eval

import pandas as pd
import chromadb
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG — edit these for your setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Dataset path inside Render/Local environment
CSV_PATH = str(DATA_DIR / "restaurant_reviews_enriched_imputed.csv.gz")

# Auto-download from GitHub Release if missing
if not os.path.exists(CSV_PATH):
    print("Downloading dataset from GitHub Releases...")
    DOWNLOAD_URL = "https://github.com/AshishNalawade0188/Restaurant-Analytics-Predictive-Intelligence-System/releases/download/v1.0.0/restaurant_reviews_enriched_imputed.csv.gz"
    urllib.request.urlretrieve(DOWNLOAD_URL, CSV_PATH)
    print("Dataset download complete!")

# Row sampling limit to remain within 512MB RAM free tier
SAMPLE_SIZE = 10000          
REBUILD_INDEX = False        # Set to False to reuse existing Chroma storage
CHROMA_PATH = str(SCRIPT_DIR / "chroma_storage")
COLLECTION_NAME = "zomato_rag"

# --- Embedding throughput + chunking config ---
EMBED_MAX_SEQ_TOKENS = 512          
CHUNK_SIZE = 400                    
CHUNK_OVERLAP = 60                  
EMBED_BATCH_SIZE = 128              

# Preprocessed CSV Column Mapping
COLUMN_MAP = {
    "name": "name",
    "location": "location",
    "rest_type": "rest_type",
    "cuisines": "cuisines",
    "dish_liked": "dish_liked",
    "cost_for_two": "approx_cost(for two people)",
    "rate": "rate",
    "votes": "votes",
    "online_order": "online_order",
    "book_table": "book_table",
    "menu_item": "menu_item",
    "reviews_list": "reviews_list",
    "listed_in_type": "listed_in(type)",
    "is_rate_imputed": "is_rate_imputed",
    "clean_review": "clean_review",
    "avg_sentiment_score": "avg_sentiment_score",
    "dominant_sentiment": "dominant_sentiment",
    "positive_ratio": "positive_ratio",
    "keywords": "keywords",
    "dish_keywords": "dish_keywords",
    "review_quality_flag": "review_quality_flag",
    "nlp_source": "nlp_source",
}

NLP_COLUMN_MAP = {
    "positive_review_count": "positive_review_count",
    "negative_review_count": "negative_review_count",
    "neutral_review_count": "neutral_review_count",
    "total_reviews_parsed": "total_reviews_parsed",
    "review_keywords": "review_keywords",
    "keywords_dish_enriched": "keywords_dish_enriched",
    "avg_review_length": "avg_review_length",
}

# ---------------------------------------------------------------------------
# SYSTEM PROMPT DEFINITION
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable, and objective food and restaurant discovery assistant for Zomato.\n"
    "Your main objective is to provide helpful, natural, and conversational responses based strictly on the retrieved context.\n\n"
    "Core Guidelines:\n"
    "1. Speak naturally like a local dining guide. Never list raw metadata keys or output raw field labels "
    "(such as 'sentiment_score', 'positive_ratio', 'nlp_source', or 'keywords:').\n"
    "2. Handle negative reviews with empathy and nuance. Instead of regurgitating raw token lists (e.g., 'found hair, kill, food thanks'), "
    "summarize core issues naturally (e.g., mentioning specific hygiene concerns or food quality feedback reported by customers).\n"
    "3. Seamlessly weave restaurant statistics (ratings, average costs, location) into flowing sentences.\n"
    "4. If certain details are missing from the context, state it naturally without sounding overly rigid or robotic."
)

# ---------------------------------------------------------------------------
# STEP 0: Model settings
# ---------------------------------------------------------------------------
USE_REAL_MODELS = True

if USE_REAL_MODELS:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("WARNING: GROQ_API_KEY is not set in environment or .env file.")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    if GROQ_API_KEY:
        Settings.llm = OpenAILike(
            model="llama-3.1-8b-instant",
            api_base="https://api.groq.com/openai/v1",
            api_key=GROQ_API_KEY,
            is_chat_model=True,
            context_window=131072,
            max_tokens=512,
        )
else:
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.llms import MockLLM
    Settings.embed_model = MockEmbedding(embed_dim=384)
    Settings.llm = MockLLM()

Settings.chunk_size = CHUNK_SIZE
Settings.chunk_overlap = CHUNK_OVERLAP
SPLITTER = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ---------------------------------------------------------------------------
# STEP 1: Load CSV (Memory Optimized)
# ---------------------------------------------------------------------------
def load_dataframe(csv_path: str) -> pd.DataFrame:
    # Only load columns actively used by the pipeline to optimize RAM
    needed_cols = set(COLUMN_MAP.values())
    
    # Read CSV with compression handling and column filter
    try:
        df = pd.read_csv(
            csv_path, 
            compression='gzip' if csv_path.endswith('.gz') else None,
            usecols=lambda c: c in needed_cols
        )
    except Exception:
        df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} rows from {csv_path}")

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
        print(f"Sampled down to {len(df)} rows to optimize cloud RAM usage.\n")

    name_col = COLUMN_MAP["name"]
    if name_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[name_col]).reset_index(drop=True)
        if before != len(df):
            print(f"Dropped {before - len(df)} rows with missing name.\n")

    return df


def col(row: pd.Series, key: str, default="", nlp=False):
    source_map = NLP_COLUMN_MAP if nlp else COLUMN_MAP
    actual_col = source_map.get(key)
    if actual_col is None or actual_col not in row.index:
        return default
    val = row[actual_col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val


def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean

# ---------------------------------------------------------------------------
# STEP 2: Parsing utilities
# ---------------------------------------------------------------------------
def parse_reviews(raw_reviews_list) -> str:
    if not raw_reviews_list or not isinstance(raw_reviews_list, str):
        return ""
    try:
        parsed = literal_eval(raw_reviews_list)
        texts = [str(r[1]).strip() for r in parsed if isinstance(r, (tuple, list)) and len(r) >= 2]
        return " ".join(texts)
    except (ValueError, SyntaxError):
        return ""


def parse_menu_items(raw_menu) -> list:
    if not raw_menu or not isinstance(raw_menu, str):
        return []
    try:
        parsed = literal_eval(raw_menu)
        return [str(m).strip() for m in parsed] if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []

# ---------------------------------------------------------------------------
# STEP 3: Narrative Document builder
# ---------------------------------------------------------------------------
def build_restaurant_document(row: pd.Series) -> Document:
    name = col(row, "name")
    rate_raw = col(row, "rate", default=None)
    rate_is_imputed = bool(col(row, "is_rate_imputed", default=False))

    clean_review = col(row, "clean_review")
    review_text = clean_review if clean_review else parse_reviews(col(row, "reviews_list"))

    menu_items = parse_menu_items(col(row, "menu_item"))
    menu_available = len(menu_items) > 0
    menu_clause = f"Notable menu highlights include {', '.join(menu_items)}." if menu_available else ""

    sentiment_label = col(row, 'dominant_sentiment')
    sentiment_sentence = f"Overall customer feedback is predominantly {sentiment_label}." if sentiment_label else ""

    raw_keywords = col(row, 'keywords')
    keyword_clause = f"Common topics and key phrases in customer feedback include: {raw_keywords}." if raw_keywords else ""

    text_content = (
        f"{name} is a {col(row, 'rest_type', 'restaurant')} located in {col(row, 'location', 'an unspecified location')}, "
        f"offering {col(row, 'cuisines', 'a variety of cuisines')}. "
        f"Popular or frequently ordered dishes include: {col(row, 'dish_liked') or 'unspecified dishes'}. "
        f"The approximate cost for two people is ₹{col(row, 'cost_for_two', 'N/A')}. "
        f"The overall user rating stands at {rate_raw or 'N/A'}. "
        f"{sentiment_sentence} {keyword_clause} {menu_clause} "
        f"Detailed customer review text: {review_text or 'No direct review text available.'}"
    )

    metadata = sanitize_metadata({
        "doc_type": "restaurant",
        "name": name,
        "location": col(row, "location", default=None),
        "cuisines": col(row, "cuisines", default=None),
        "rest_type": col(row, "rest_type", default=None),
        "cost_for_two": col(row, "cost_for_two", default=None),
        "rating": rate_raw,
        "rating_is_imputed": rate_is_imputed,
        "online_order": col(row, "online_order", default=None),
        "book_table": col(row, "book_table", default=None),
        "menu_available": menu_available,
        "sentiment_score": col(row, "avg_sentiment_score", default=None),
        "sentiment_label": sentiment_label,
        "positive_ratio": col(row, "positive_ratio", default=None),
        "keywords": raw_keywords,
        "dish_keywords": col(row, "dish_keywords", default=None),
        "review_quality": col(row, "review_quality_flag", default=None),
        "nlp_source": col(row, "nlp_source", default=None),
    })
    return Document(text=text_content, metadata=metadata)


DEFAULT_FAQS = [
    {"question": "How do I cancel my Zomato order?",
     "answer": "Go to Orders, select the active order, and tap Cancel. Refunds are processed within 5-7 business days."},
    {"question": "How does Zomato table booking work?",
     "answer": "Search a restaurant, check slot availability, and confirm. Some restaurants require a refundable deposit."},
]


def build_faq_document(faq: dict) -> Document:
    return Document(text=f"Q: {faq['question']} A: {faq['answer']}",
                     metadata={"doc_type": "faq", "question": faq["question"]})

# ---------------------------------------------------------------------------
# STEP 4: Build the index
# ---------------------------------------------------------------------------
def build_index() -> VectorStoreIndex:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if not REBUILD_INDEX:
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
            existing_count = chroma_collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            print(f"Loaded existing index with {existing_count} documents from {CHROMA_PATH}.\n")
            return index

    df = load_dataframe(CSV_PATH)
    restaurant_docs = [build_restaurant_document(row) for _, row in df.iterrows()]
    faq_docs = [build_faq_document(f) for f in DEFAULT_FAQS]
    all_documents = restaurant_docs + faq_docs

    if REBUILD_INDEX:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        all_documents,
        storage_context=storage_context,
        transformations=[SPLITTER],
        show_progress=True,
    )
    return index

# ---------------------------------------------------------------------------
# STEP 5: Interactive chat loop
# ---------------------------------------------------------------------------
def run_chat(index: VectorStoreIndex):
    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=3,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.4,
    )
    print("Zomato RAG chatbot ready. Type 'exit' to quit, 'sources' to see last retrieval.\n")
    last_response = None
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "sources" and last_response is not None:
            for n in last_response.source_nodes:
                print(f"  - [{n.metadata.get('doc_type')}] {n.metadata.get('name', n.metadata.get('question'))} (score={n.score:.3f})")
            continue
        last_response = chat_engine.chat(query)
        print(f"Bot: {last_response}\n")

if __name__ == "__main__":
    idx = build_index()
    run_chat(idx)
