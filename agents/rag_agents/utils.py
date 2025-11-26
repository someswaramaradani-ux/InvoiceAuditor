# agents/rag_agents/utils.py
import os
import json
from pathlib import Path
from typing import List, Dict

DATA_DIR = Path("data")
REPORTS_DIR = DATA_DIR / "reports"
CHUNKS_DIR = DATA_DIR / "rag_chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def list_report_files():
    return sorted([p for p in REPORTS_DIR.glob("*.report.json")])
