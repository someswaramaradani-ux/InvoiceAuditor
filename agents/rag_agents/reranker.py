# agents/rag_agents/reranker.py
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from agents.rag_agents.retriever import retrieve, load_index
from agents.rag_agents.utils import load_json
import numpy as np
from pathlib import Path

# This reranker uses a pair-encoding trick with a model that supports cross-encoding.
# For production, use a proper cross-encoder model (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2').
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    RERANKER = SentenceTransformer(RERANK_MODEL_NAME)
except Exception:
    # fallback to bi-encoder if cross-encoder unavailable
    RERANKER = SentenceTransformer("all-MiniLM-L6-v2")

CHUNKS_PATH = Path("data/rag_chunks/chunks.json")

def load_chunks():
    return load_json(CHUNKS_PATH)

def rerank(query: str, retrieved_meta: List[Dict], top_k: int = 5):
    # load raw chunks to access text content
    chunks = load_chunks()
    # build candidate list (meta -> text)
    candidates = []
    for r in retrieved_meta:
        # find chunk by chunk_id
        cid = r["meta"]["chunk_id"]
        found = next((c for c in chunks if c["meta"]["chunk_id"] == cid), None)
        if found:
            candidates.append({"meta": r["meta"], "text": found["text"], "score": r["score"]})
    if not candidates:
        return []
    pair_inputs = [(query, c["text"]) for c in candidates]
    # cross-encoder expects list of (query, passage) pairs
    try:
        scores = RERANKER.predict(pair_inputs)  # for cross-encoder, `.predict` returns score
    except Exception:
        # fallback: compute dot product of embeddings
        q_emb = RERANKER.encode([query], convert_to_numpy=True)
        p_embs = RERANKER.encode([c["text"] for c in candidates], convert_to_numpy=True)
        # normalize and dot
        from numpy.linalg import norm
        q_emb_n = q_emb / (norm(q_emb, axis=1, keepdims=True) + 1e-12)
        p_embs_n = p_embs / (norm(p_embs, axis=1, keepdims=True) + 1e-12)
        scores = (p_embs_n @ q_emb_n.T).squeeze().tolist()
    # attach scores and sort
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]

def run(params=None, context=None):
    query = (params or {}).get("query", "What discrepancies are present?")
    top_k = int((params or {}).get("top_k", 5))
    retriever_out = (params or {}).get("retrieved")  # or call retrieve internally
    if retriever_out:
        retrieved = retriever_out
    else:
        retrieved = retrieve(query, top_k=top_k)
    reranked = rerank(query, retrieved, top_k=top_k)
    return {"query": query, "reranked": reranked}
