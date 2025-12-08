# scripts/build_faiss_index.py
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

REPORTS = Path("data/reports")
INDEX_DIR = Path("data/faiss")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "all-MiniLM-L6-v2"

def load_reports():
    docs = []
    paths = []
    for p in REPORTS.glob("*.report.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            text = json.dumps(j, ensure_ascii=False)
            docs.append(text)
            paths.append(str(p.resolve()))
        except Exception as e:
            print("skip", p, e)
    return paths, docs

def build_index(paths, texts):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, str(INDEX_DIR / "invoices.index"))
    (INDEX_DIR / "meta.json").write_text(json.dumps(paths, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote index and meta:", INDEX_DIR)

if __name__ == "__main__":
    paths, texts = load_reports()
    if not texts:
        print("No reports found in data/reports. Place .report.json files there.")
        raise SystemExit(1)
    build_index(paths, texts)
