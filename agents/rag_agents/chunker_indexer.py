# agents/rag_agents/chunker_indexer.py
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from agents.rag_agents.utils import list_report_files, CHUNKS_DIR, save_json

MODEL_NAME = "all-MiniLM-L6-v2"   # small and fast; swap for production
EMB_MODEL = SentenceTransformer(MODEL_NAME)

INDEX_DIR = Path("data/faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = INDEX_DIR / "invoices.index"
META_PATH = INDEX_DIR / "meta.json"
CHUNK_SIZE = 600          # characters per chunk (tune)
CHUNK_OVERLAP = 120       # characters overlap between chunks

def text_to_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.replace("\r", " ")
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        start = end - overlap

def build_corpus_from_reports():
    docs = list_report_files()
    corpus = []
    for doc_path in docs:
        j = json.load(open(doc_path, "r", encoding="utf-8"))
        text = json.dumps(j, ensure_ascii=False)  # incorporate structured fields and problems
        for i, chunk in enumerate(text_to_chunks(text)):
            meta = {
                "source_report": str(doc_path),
                "chunk_id": f"{doc_path.stem}__{i}",
                "offset": i
            }
            corpus.append({"text": chunk, "meta": meta})
    return corpus

def run(params=None, context=None):
    corpus = build_corpus_from_reports()
    if not corpus:
        return {"indexed": 0, "message": "no reports found to index"}
    texts = [c["text"] for c in corpus]
    embs = EMB_MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    d = embs.shape[1]

    index = faiss.IndexFlatIP(d)  # using inner product on normalized vectors (cosine)
    # normalize embeddings to unit vectors for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embs)
    index.add(embs.astype("float32"))
    faiss.write_index(index, str(INDEX_PATH))

    # Persist metadata (list of meta dicts in the same order)
    meta = [c["meta"] for c in corpus]
    save_json(META_PATH, meta)

    # Save raw chunks for debugging & display
    save_json(CHUNKS_DIR / "chunks.json", corpus)

    return {"indexed": len(corpus)}
