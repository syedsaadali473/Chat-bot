# ── config.py ──────────────────────────────────────────────────────────────
import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDWqfh_I6ZzOKB3rxfwOUQjNbE0jIM9eDY")
API_VERSION = "v1"
# Gemini models (new SDK)
EMBEDDING_MODEL = "gemini-embedding-001"   # no "models/" prefix needed
CHAT_MODEL      = "gemini-robotics-er-1.5-preview"

# RAG settings
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 5

# Paths
DATA_DIR        = "data"
VECTORSTORE_DIR = "vectorstore"
COLLECTION_NAME = "audit_docs"
