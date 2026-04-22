# ── chatbot.py ─────────────────────────────────────────────────────────────
# Usage: python chatbot.py

from google import genai
from google.genai import types
import chromadb
from config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL,
    TOP_K, VECTORSTORE_DIR, COLLECTION_NAME,
)

# ── 1. Configure Gemini ────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── 2. Load ChromaDB collection ───────────────────────────────────────────
def load_collection():
    chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        print(f"[INFO] Loaded collection '{COLLECTION_NAME}' ({collection.count()} chunks)\n")
        return collection
    except Exception:
        print("[ERROR] Vector store not found. Run 'python ingest.py' first.")
        exit(1)

# ── 3. Retrieve top-K relevant chunks ─────────────────────────────────────
def retrieve(query, collection, top_k):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    query_embedding = result.embeddings[0].values

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "source": meta["source"], "distance": dist})

    return chunks

# ── 4. Build prompt ────────────────────────────────────────────────────────
def build_prompt(query, chunks):
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] (Source: {chunk['source']})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    return f"""You are an internal audit assistant. Answer the question below using ONLY the context provided.
If the context does not contain enough information, say so clearly. Do not make up facts.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

# ── 5. Chat loop ───────────────────────────────────────────────────────────
def chat():
    collection = load_collection()
    print("=" * 55)
    print("  Audit RAG Chatbot — type 'exit' to quit")
    print("=" * 55)

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        chunks = retrieve(query, collection, TOP_K)
        sources = list({c["source"] for c in chunks})
        print(f"\n[Sources: {', '.join(sources)}]")

        prompt = build_prompt(query, chunks)
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
        )
        print(f"\nBot: {response.text}")

# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chat()
