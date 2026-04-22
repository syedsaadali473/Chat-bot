# ── ingest.py ──────────────────────────────────────────────────────────────
# Run once to build the vector store.
# Usage: python ingest.py

import os
import time
from google import genai
from google.genai import types
import chromadb
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
    DATA_DIR, VECTORSTORE_DIR, COLLECTION_NAME,
)

# ── 1. Configure Gemini ────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── 2. Load all .txt files from data/ ─────────────────────────────────────
def load_texts(data_dir):
    documents = []
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"[INFO] Created '{data_dir}/' folder. Add your .txt files there and re-run.")
        return documents

    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                documents.append({"filename": fname, "text": text})
                print(f"  [LOADED] {fname}  ({len(text):,} chars)")

    return documents

# ── 3. Chunk text ──────────────────────────────────────────────────────────
def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ── 4. Embed texts via new Gemini SDK ─────────────────────────────────────
def embed_texts(texts):
    embeddings = []
    for i, text in enumerate(texts):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings.append(result.embeddings[0].values)
        if (i + 1) % 10 == 0:
            print(f"    Embedded {i + 1}/{len(texts)} chunks...")
            time.sleep(1)
    return embeddings

# ── 5. Store in ChromaDB ───────────────────────────────────────────────────
def build_vectorstore(documents):
    chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print("[INFO] Cleared existing collection.")
    except Exception:
        pass

    collection = chroma_client.create_collection(COLLECTION_NAME)

    all_chunks, all_ids, all_metas = [], [], []
    chunk_counter = 0

    for doc in documents:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_ids.append(f"chunk_{chunk_counter}")
            all_metas.append({"source": doc["filename"]})
            chunk_counter += 1

    print(f"\n[INFO] Total chunks to embed: {len(all_chunks)}")
    embeddings = embed_texts(all_chunks)

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )
    print(f"[DONE] Stored {len(all_chunks)} chunks in ChromaDB → '{VECTORSTORE_DIR}/'")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ingesting documents ===\n")
    docs = load_texts(DATA_DIR)
    if not docs:
        print("[WARN] No documents found. Add .txt files to the 'data/' folder.")
    else:
        build_vectorstore(docs)
