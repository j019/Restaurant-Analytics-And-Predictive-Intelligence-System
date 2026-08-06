"""
Zomato RAG Chatbot — Refactored CSV-driven Pipeline (Cloud Memory-Optimized)
=============================================================================
- Low RAM Footprint: Imports heavy libraries (Pandas, Chroma, FastEmbed) inside 
  functions to keep idle startup RAM under 100 MB for Render Free Tier.
- FastEmbed Integration: Eliminates PyTorch memory overhead.
- Narrative Document Builder: Constructs human-readable narrative sentences.
"""

import os
import math
import urllib.request
from pathlib import Path
from ast import literal_eval
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = str(DATA_DIR / "restaurant_reviews_enriched_imputed.csv.gz")
CHROMA_PATH = str(SCRIPT_DIR / "chroma_storage")
COLLECTION_NAME = "zomato_rag"

# Auto-download from GitHub Release if missing
if not os.path.exists(CSV_PATH):
    print("Downloading dataset from GitHub Releases...")
    DOWNLOAD_URL = "https://github.com/AshishNalawade0188/Restaurant-Analytics-Predictive-Intelligence-System/releases/download/v1.0.0/restaurant_reviews_enriched_imputed.csv.gz"
    urllib.request.urlretrieve(DOWNLOAD_URL, CSV_PATH)
    print("Dataset download complete!")

# Sample size limit for RAM conservation
SAMPLE_SIZE = 5000
REBUILD_INDEX = False

# System Prompt Definition
SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable, and objective food and restaurant discovery assistant for Zomato.\n"
    "Your main objective is to provide helpful, natural, and conversational responses based strictly on the retrieved context.\n\n"
    "Core Guidelines:\n"
    "1. Speak naturally like a local dining guide. Never list raw metadata keys or output raw field labels.\n"
    "2. Handle negative reviews with empathy and nuance.\n"
    "3. Seamlessly weave restaurant statistics (ratings, average costs, location) into flowing sentences.\n"
    "4. If certain details are missing from the context, state it naturally without sounding robotic."
)

COLUMN_MAP = {
    "name": "name",
    "location": "location",
    "rest_type": "rest_type",
    "cuisines": "cuisines",
    "dish_liked": "dish_liked",
    "cost_for_two": "approx_cost(for two people)",
    "rate": "rate",
    "online_order": "online_order",
    "book_table": "book_table",
    "menu_item": "menu_item",
    "reviews_list": "reviews_list",
    "clean_review": "clean_review",
    "dominant_sentiment": "dominant_sentiment",
    "keywords": "keywords",
}


def sanitize_metadata(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if isinstance(v, float) and math.isnan(v):
            clean[k] = None
        else:
            clean[k] = v
    return clean


def parse_reviews(raw_reviews_list) -> str:
    if not raw_reviews_list or not isinstance(raw_reviews_list, str):
        return ""
    try:
        parsed = literal_eval(raw_reviews_list)
        texts = [str(r[1]).strip() for r in parsed if isinstance(r, (tuple, list)) and len(r) >= 2]
        return " ".join(texts)
    except (ValueError, SyntaxError):
        return ""


def build_restaurant_document(row, Document_cls) -> object:
    def col(key, default=""):
        actual_col = COLUMN_MAP.get(key)
        if actual_col is None or actual_col not in row.index:
            return default
        val = row[actual_col]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return val

    name = col("name")
    rate_raw = col("rate", default="N/A")
    clean_review = col("clean_review")
    review_text = clean_review if clean_review else parse_reviews(col("reviews_list"))

    sentiment_label = col("dominant_sentiment")
    sentiment_sentence = f"Overall customer feedback is predominantly {sentiment_label}." if sentiment_label else ""

    raw_keywords = col("keywords")
    keyword_clause = f"Common topics in customer feedback include: {raw_keywords}." if raw_keywords else ""

    text_content = (
        f"{name} is a {col('rest_type', 'restaurant')} located in {col('location', 'an unspecified location')}, "
        f"offering {col('cuisines', 'a variety of cuisines')}. "
        f"Popular dishes include: {col('dish_liked') or 'unspecified dishes'}. "
        f"Approximate cost for two: ₹{col('cost_for_two', 'N/A')}. "
        f"Overall user rating: {rate_raw}. "
        f"{sentiment_sentence} {keyword_clause} "
        f"Detailed reviews: {review_text or 'No direct review text available.'}"
    )

    metadata = sanitize_metadata({
        "doc_type": "restaurant",
        "name": name,
        "location": col("location", default=None),
        "cuisines": col("cuisines", default=None),
        "rating": rate_raw,
    })
    return Document_cls(text=text_content, metadata=metadata)


# ---------------------------------------------------------------------------
# LAZY INDEX BUILDER
# ---------------------------------------------------------------------------
def build_index():
    """Builds or loads the vector index dynamically with lazy imports."""
    import pandas as pd
    import chromadb
    from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.llms.openai_like import OpenAILike
    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable is missing.")
        return None

    # Global Settings Setup (Lightweight FastEmbed)
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = OpenAILike(
        model="llama-3.1-8b-instant",
        api_base="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
        is_chat_model=True,
        context_window=131072,
        max_tokens=512,
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Return existing index if available and REBUILD_INDEX is False
    if not REBUILD_INDEX:
        try:
            chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
            if chroma_collection.count() > 0:
                print(f"Loading existing Chroma collection '{COLLECTION_NAME}' ({chroma_collection.count()} docs)...")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                return VectorStoreIndex.from_vector_store(vector_store)
        except Exception:
            pass

    # Build Index from CSV
    print(f"Building fresh vector index from {CSV_PATH}...")
    needed_cols = set(COLUMN_MAP.values())
    
    try:
        df = pd.read_csv(
            CSV_PATH, 
            compression='gzip' if CSV_PATH.endswith('.gz') else None,
            usecols=lambda c: c in needed_cols
        )
    except Exception:
        df = pd.read_csv(CSV_PATH)

    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    name_col = COLUMN_MAP["name"]
    if name_col in df.columns:
        df = df.dropna(subset=[name_col]).reset_index(drop=True)

    documents = [build_restaurant_document(row, Document) for _, row in df.iterrows()]

    if REBUILD_INDEX:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=60)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    return index


if __name__ == "__main__":
    idx = build_index()
    if idx:
        chat_engine = idx.as_chat_engine(
            chat_mode="condense_plus_context",
            similarity_top_k=3,
            system_prompt=SYSTEM_PROMPT,
            temperature=0.4,
        )
        print("Chatbot ready! Type 'exit' to quit.")
        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            print(f"Bot: {chat_engine.chat(query)}\n")
