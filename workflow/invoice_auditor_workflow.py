#!/usr/bin/env python3
"""
Simple workflow runner for the Invoice Auditor — NO LANGGRAPH REQUIRED.

Nodes are executed in the exact order defined in invoice_auditor_workflow.yaml:
  monitor → extractor → translator → validator → business_validator → reporter
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import time
import json
import traceback
from typing import Optional, Dict, Any
import subprocess
import signal

# -----------------------------------------------------------------------------
# Ensure imports from repo root
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Spec path from your uploaded file
SPEC_PATH = "/mnt/data/AI Invoice Auditor Agentic AI.docx"

# Node module map
NODE_IMPORTS = {
    "monitor": "agents.monitor_node",
    "extractor": "agents.extractor_node",
    "translator": "agents.translation_node",
    "validator": "agents.validation_node",
    "business_validator": "agents.business_validation_node",
    "reporter": "agents.reporting_node"
}

# ------------------------------------------------------------------------
# Importing agent node modules
# ------------------------------------------------------------------------
def import_node(module_name: str):
    try:
        mod = __import__(module_name, fromlist=["*"])
        return mod
    except Exception as e:
        print(f"[import_node] ERROR: Failed to import {module_name}: {e}")
        return None

MODULES = {k: import_node(v) for k, v in NODE_IMPORTS.items()}

# ------------------------------------------------------------------------
# Call a node's run() method safely
# ------------------------------------------------------------------------
def call_node(node_key: str, params: Optional[Dict[str, Any]] = None):
    params = params or {}
    mod = MODULES.get(node_key)
    if mod is None:
        raise RuntimeError(f"Module for node '{node_key}' not found.")

    run_fn = getattr(mod, "run", None)
    if not callable(run_fn):
        raise RuntimeError(f"Node '{node_key}' has no run(params, context) function.")

    print(f"\n[workflow] ▶ Running node: {node_key}")
    try:
        result = run_fn(params, {"spec_path": SPEC_PATH})
        print(f"[workflow] ✔ Node '{node_key}' complete.")
        return result
    except Exception as e:
        print(f"[workflow] ✘ Node '{node_key}' failed: {e}")
        traceback.print_exc()
        raise

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

# ------------------------------------------------------------------------
# Manual sequential execution of all nodes
# ------------------------------------------------------------------------
def manual_run() -> Dict[str, Any]:
    results = {}
    erp_proc = None
    try:
        #erp_proc = start_mock_erp()
        #print("[demo] Waiting a couple seconds for ERP to be ready...")
        #time.sleep(2.0)

        # 1) monitor
        results["monitor"] = call_node("monitor",
            {"watch_path": "data/incoming", "processed_path": "data/processed"}
        )

        # 2) extractor
        results["extractor"] = call_node("extractor")

        # 3) translator
        results["translator"] = call_node("translator")

        # 4) validator
        results["validator"] = call_node("validator")

        # 5) business_validator
        results["business_validator"] = call_node("business_validator")

        # 6) reporter
        results["reporter"] = call_node("reporter")

        print("\n[workflow] 🎉 Completed full pipeline.")
        return results
    except Exception as e:
        print("[demo] ERROR:", e)
        raise

# ------------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------------
def main(run_once=True, schedule_interval=None):
    print(f"run_once: {run_once}")
    if run_once:
        out = manual_run()
        print(json.dumps(out, indent=2))
        return

    # schedule mode
    print(f"[workflow] Running every {schedule_interval} seconds. Press Ctrl+C to stop.")
    while True:
        try:
            manual_run()
        except Exception as e:
            print("[workflow] ERROR in scheduled run:", e)
        time.sleep(schedule_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-once", action="store_true", help="Run the workflow once and exit.")
    parser.add_argument("--schedule", type=int, default=300, help="Run every N seconds.")
    args = parser.parse_args()

    if args.run_once:
        main(run_once=True)
    else:
        main(run_once=False, schedule_interval=args.schedule)
