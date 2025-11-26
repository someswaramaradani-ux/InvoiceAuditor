#!/usr/bin/env python3
"""
End-to-end demo runner for the LangGraph - Invoice Auditor Workflow & Agents.

What it does (in order):
  1. Optionally starts the mock ERP (FastAPI) in a background subprocess.
  2. Ensures data folders exist and copies a sample invoice into data/incoming (if not present).
  3. Invokes each node in the pipeline in sequence:
       monitor -> extractor -> translator -> validator -> business_validator -> reporter -> indexer -> rag_qa(retrieve)
  4. Prints locations of generated reports, FAISS index, and a QA sample output.
  5. Stops mock ERP subprocess if it started it.

Usage:
  python scripts/run_demo_pipeline.py [--start-erp] [--sample-invoice PATH]

Notes:
  - This script expects the repository layout described in the canvas document:
      agents/, agents/rag_agents/, mock_erp/, configs/, data/
  - It calls the `run(params, context)` functions from each node file (file-based handoff).
  - For real LLMs, set up `agents/litellm_client.py` before running (or the nodes will use their mocks).
  - SPEC_PATH references your uploaded spec file for traceability:
      /mnt/data/AI Invoice Auditor Agentic AI.docx
"""

import argparse
import subprocess
import time
import shutil
import json
from pathlib import Path
import sys
import signal

# Default spec path (uploaded file) — per your request this local path is exposed here.
SPEC_PATH = Path("/mnt/data/AI Invoice Auditor Agentic AI.docx")
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Paths used by nodes (match node code)
BASE = Path.cwd()
DATA_INCOMING = BASE / "data" / "incoming"
DATA_PROCESSED = BASE / "data" / "processed"
DATA_REPORTS = BASE / "data" / "reports"
FAISS_DIR = BASE / "data" / "faiss"

# Node modules to import (these must be available in pythonpath)
NODE_MONITOR = "agents.monitor_node"
NODE_EXTRACTOR = "agents.extractor_node"
NODE_TRANSLATOR = "agents.translation_node"
NODE_VALIDATOR = "agents.validation_node"
NODE_BUSINESS = "agents.business_validation_node"
NODE_REPORTER = "agents.reporting_node"
NODE_INDEXER = "agents.rag_agents.indexing_node"
NODE_RAG_RETRIEVAL = "agents.rag_agents.retrieval_node"

def ensure_dirs():
    for d in [DATA_INCOMING, DATA_PROCESSED, DATA_REPORTS, FAISS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def start_mock_erp():
    """Start mock ERP app via uvicorn in background subprocess. Returns subprocess handle."""
    print("[demo] Starting mock ERP (uvicorn mock_erp.app:app --port 8000)")
    # Use env copy so local Python path is available.
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "mock_erp.app:app", "--reload", "--port", "8000"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # give it a moment to start
    time.sleep(1.2)
    return proc

def stop_process(proc):
    if not proc:
        return
    try:
        proc.send_signal(signal.SIGINT)
        time.sleep(0.5)
        proc.kill()
    except Exception:
        pass

def copy_sample_invoice(sample_source: Path):
    """
    Copy sample invoice (PDF/TXT/PNG/DOCX) into data/incoming if the folder is empty.
    If provided sample_source doesn't exist, create a small .txt invoice sample.
    """
    incoming_files = list(DATA_INCOMING.glob("*"))
    if incoming_files:
        print("[demo] data/incoming already has files; skipping sample copy.")
        return

    if sample_source and sample_source.exists():
        dest = DATA_INCOMING / sample_source.name
        print(f"[demo] Copying provided sample invoice {sample_source} -> {dest}")
        shutil.copy(sample_source, dest)
    else:
        # create a small text invoice
        dest = DATA_INCOMING / "demo_invoice_1.txt"
        text = """Factura No: INV-2025-1001
Fecha: 2025-11-20
Proveedor: ACME Co. Ltd.
Linea 1: SKU-1 2 @ 25.00 = 50.00
Linea 2: SKU-2 1 @ 50.00 = 50.00
TOTAL: 100.00 USD
"""
        dest.write_text(text, encoding="utf-8")
        print(f"[demo] Generated sample invoice at {dest}")

def import_and_run(module_name: str, params: dict = None):
    """
    Import the module and call run(params, context) if present.
    Returns the result of run(...) or raw module if run not present.
    """
    params = params or {}
    try:
        mod = __import__(module_name, fromlist=["*"])
    except Exception as e:
        raise RuntimeError(f"Failed to import {module_name}: {e}")
    run_fn = getattr(mod, "run", None)
    if callable(run_fn):
        print(f"[demo] Running node: {module_name}.run()")
        res = run_fn(params, {})
        print(f"[demo] Node {module_name} result: {type(res).__name__}")
        return res
    else:
        print(f"[demo] Module {module_name} has no run(params,context) — returning module")
        return mod

def pretty_print_json(fn_path):
    p = Path(fn_path)
    if not p.exists():
        print(f"[demo] Not found: {p}")
        return
    try:
        data = json.load(open(p, "r", encoding="utf-8"))
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print("[demo] Could not parse JSON -- printing raw")
        print(p.read_text(encoding="utf-8")[:2000])

def main(start_erp=False, sample_invoice=None):
    ensure_dirs()
    erp_proc = None
    try:
        if start_erp:
            erp_proc = start_mock_erp()
            print("[demo] Waiting a couple seconds for ERP to be ready...")
            time.sleep(2.0)

        # copy / generate a sample invoice into data/incoming
        copy_sample_invoice(sample_invoice)

        # 1) monitor node — lists files in incoming and writes monitor_output.json
        monitor_out = import_and_run(NODE_MONITOR, params={"watch_path": str(DATA_INCOMING), "processed_path": str(DATA_PROCESSED)})

        # 2) extractor node — reads monitor_output.json and produces extracted json(s)
        extractor_out = import_and_run(NODE_EXTRACTOR)

        # 3) translation node — translate & produce translated files (calls LLM wrapper)
        translator_out = import_and_run(NODE_TRANSLATOR)

        # 4) validation node — perform totals & rules validations
        validator_out = import_and_run(NODE_VALIDATOR)

        # 5) business validator — call mock ERP to cross-check PO/lines
        biz_out = import_and_run(NODE_BUSINESS)

        # 6) reporting node — produce reports under data/reports
        reporter_out = import_and_run(NODE_REPORTER)
        print("[demo] Reports written to:", DATA_REPORTS)

        # Print summary and sample outputs
        print("\n=== DEMO SUMMARY ===")
        print("Spec file used (for reference):", SPEC_PATH)
        print("Incoming files:", list(DATA_INCOMING.glob("*")))
        print("Processed JSONs:", list(DATA_PROCESSED.glob("*.json"))[:10])
        print("Reports:", list(DATA_REPORTS.glob("*.report.json")))
        print("FAISS index dir:", FAISS_DIR)
        print("Sample retriever output:")
        

        # show one report JSON (if any)
        reports = list(DATA_REPORTS.glob("*.report.json"))
        if reports:
            print("\nOpening first generated report JSON:")
            pretty_print_json(reports[0])

        print("\nDemo finished successfully.")
    except Exception as e:
        print("[demo] ERROR:", e)
        raise
    finally:
        if erp_proc:
            print("[demo] Stopping mock ERP process...")
            stop_process(erp_proc)
            print("[demo] mock ERP stopped.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-erp", action="store_true", help="If set, start the mock ERP (uvicorn) in the background.")
    ap.add_argument("--sample-invoice", type=str, default=None, help="Path to a sample invoice file to copy to data/incoming. If omitted, demo invoice is generated.")
    args = ap.parse_args()
    sample = Path(args.sample_invoice) if args.sample_invoice else None
    main(start_erp=args.start_erp, sample_invoice=sample)
