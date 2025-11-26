# agents/rag_agents/retriever.py
import numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from agents.rag_agents.utils import load_json
from typing import List, Dict

INDEX_PATH = Path("data/faiss/invoices.index")
META_PATH = Path("data/faiss/meta.json")
MODEL_NAME = "all-MiniLM-L6-v2"
EMB_MODEL = SentenceTransformer(MODEL_NAME)

def load_index():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Index or meta not found. Run chunker_indexer first.")
    index = faiss.read_index(str(INDEX_PATH))
    meta = load_json(META_PATH)
    return index, meta

def retrieve(query: str, top_k: int = 5):
    index, meta = load_index()
    q_emb = EMB_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb.astype("float32"), k=top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        entry_meta = meta[idx]
        results.append({"score": float(score), "meta": entry_meta})
    return results

def run(params=None, context=None):
    query = (params or {}).get("query", "What discrepancies are present?")
    top_k = int((params or {}).get("top_k", 5))
    try:
        results = retrieve(query, top_k=top_k)
    except FileNotFoundError as e:
        return {"error": str(e)}
    return {"query": query, "results": results}
