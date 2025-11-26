#!/usr/bin/env python3
"""
workflow/invoice_auditor_workflow.py

Modern LangGraph Python workflow for the Invoice Auditor pipeline.

- Builds a typed StateGraph (Pydantic state model) and wires nodes that call your
  existing file-based agent modules under `agents/` and `agents/rag_agents/`.
- Falls back to a safe manual orchestration if the installed langgraph API is
  not available or incompatible.
- Preserves the uploaded spec path (local) for traceability.

Save as: workflow/invoice_auditor_workflow.py
Run:       python workflow/invoice_auditor_workflow.py --run-once
"""

from __future__ import annotations
import sys
from pathlib import Path
import argparse
import time
import json
import traceback
from typing import Optional, Dict, Any

# ensure repo root is on sys.path so imports like 'agents.*' work when running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Uploaded spec path (local) — your system will transform this to a URL as needed
SPEC_PATH = "/mnt/data/AI Invoice Auditor Agentic AI.docx"

# Node module map (module paths must exist in repo)
NODE_IMPORTS = {
    "monitor": "agents.monitor_node",
    "extractor": "agents.extractor_node",
    "translator": "agents.translation_node",
    "validator": "agents.validation_node",
    "business_validator": "agents.business_validation_node",
    "reporter": "agents.reporting_node",
}

def import_node(module_name: str):
    try:
        mod = __import__(module_name, fromlist=["*"])
        return mod
    except Exception as e:
        print(f"[import_node] Failed to import {module_name}: {e}")
        return None

# Load modules dictionary (lazy-load also possible; we import now so missing modules are visible early)
MODULES: Dict[str, Optional[object]] = {name: import_node(modpath) for name, modpath in NODE_IMPORTS.items()}

def call_node(node_key: str, params: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    Safely call the node's run(params, context) function.
    Raises RuntimeError if module or run() is not available.
    """
    params = params or {}
    context = context or {"spec_path": SPEC_PATH}
    mod = MODULES.get(node_key)
    if mod is None:
        raise RuntimeError(f"Module for node '{node_key}' not available (expected {NODE_IMPORTS.get(node_key)}).")
    run_fn = getattr(mod, "run", None)
    if not callable(run_fn):
        raise RuntimeError(f"Node module '{mod.__name__}' has no callable run(params, context).")
    print(f"[workflow] Calling node: {node_key} -> {mod.__name__}.run()")
    try:
        res = run_fn(params, context)
        print(f"[workflow] Node {node_key} completed. Result type: {type(res).__name__}")
        return res
    except Exception as e:
        print(f"[workflow] Node {node_key} raised exception: {e}")
        traceback.print_exc()
        raise

# ---------------------------
# Define Typed State for StateGraph
# ---------------------------
try:
    from pydantic import BaseModel
except Exception:
    # pydantic not installed: define a minimal fallback class so manual orchestration can still run.
    BaseModel = object

class InvoiceState(BaseModel):  # type: ignore[misc]
    monitor_output: Optional[Dict[str, Any]] = None
    extractor_output: Optional[Dict[str, Any]] = None
    translator_output: Optional[Dict[str, Any]] = None
    validator_output: Optional[Dict[str, Any]] = None
    business_validator_output: Optional[Dict[str, Any]] = None
    reporter_output: Optional[Dict[str, Any]] = None


# ---------------------------
# Try to use modern LangGraph StateGraph API
# ---------------------------
USE_LANGGRAPH = False
StateGraph = None
END = None
try:
    # modern LangGraph exposes StateGraph in langgraph.graph (may vary by version)
    from langgraph.graph import StateGraph, END  # type: ignore
    USE_LANGGRAPH = True
except Exception:
    USE_LANGGRAPH = False

def build_and_run_stategraph_once() -> Dict[str, Any]:
    """
    Build and run a LangGraph StateGraph if available; otherwise fall back to manual orchestration.
    Returns a results dict summarizing outputs.
    """
    if not USE_LANGGRAPH:
        print("[workflow] LangGraph StateGraph API not available. Using manual orchestration.")
        return manual_orchestration_run()

    print("[workflow] Building StateGraph with InvoiceState")
    graph = StateGraph(InvoiceState)  # type: ignore

    # Node wrappers — each receives and returns InvoiceState
    @graph.nodes
    def monitor_node(state: InvoiceState) -> InvoiceState:
        res = call_node("monitor", params={"watch_path": "data/incoming", "processed_path": "data/processed"})
        # store result (as-is) into state; State model will accept dicts
        state.monitor_output = res
        return state

    @graph.nodes
    def extractor_node(state: InvoiceState) -> InvoiceState:
        res = call_node("extractor")
        state.extractor_output = res
        return state

    @graph.nodes
    def translator_node(state: InvoiceState) -> InvoiceState:
        res = call_node("translator")
        state.translator_output = res
        return state

    @graph.nodes
    def validator_node(state: InvoiceState) -> InvoiceState:
        res = call_node("validator")
        state.validator_output = res
        return state

    @graph.nodes
    def business_validator_node(state: InvoiceState) -> InvoiceState:
        res = call_node("business_validator")
        state.business_validator_output = res
        return state

    @graph.nodes
    def reporter_node(state: InvoiceState) -> InvoiceState:
        res = call_node("reporter")
        state.reporter_output = res
        return state


    # Wire edges in the same order as the YAML
    graph.add_edge("monitor_node", "extractor_node")
    graph.add_edge("extractor_node", "translator_node")
    graph.add_edge("translator_node", "validator_node")
    graph.add_edge("validator_node", "business_validator_node")
    graph.add_edge("business_validator_node", "reporter_node")
    graph.add_edge("reporter_node", END)
    
    print("[workflow] Compiling StateGraph...")
    try:
        compiled = graph.compile()
        print("[workflow] Running compiled StateGraph...")
        out_state = compiled.run({})  # initial empty state
        # convert to serializable summary
        summary = {
            "monitor": getattr(out_state, "monitor_output", None),
            "extractor": getattr(out_state, "extractor_output", None),
            "translator": getattr(out_state, "translator_output", None),
            "validator": getattr(out_state, "validator_output", None),
            "business_validator": getattr(out_state, "business_validator_output", None),
            "reporter": getattr(out_state, "reporter_output", None),
        }
        return summary
    except Exception as e:
        print("[workflow] Error compiling/running StateGraph:", e)
        traceback.print_exc()
        print("[workflow] Falling back to manual orchestration.")
        return manual_orchestration_run()

def manual_orchestration_run() -> Dict[str, Any]:
    """
    Sequentially call each node's run() in the order defined in the YAML.
    This fallback works even without langgraph installed.
    """
    context = {"spec_path": SPEC_PATH}
    results: Dict[str, Any] = {}
    # 1) monitor
    try:
        results["monitor"] = call_node("monitor", params={"watch_path": "data/incoming", "processed_path": "data/processed"}, context=context)
    except Exception as e:
        print("[workflow] monitor failed:", e)
        raise

    # 2) extractor
    results["extractor"] = call_node("extractor", context=context)

    # 3) translator
    results["translator"] = call_node("translator", context=context)

    # 4) validator
    results["validator"] = call_node("validator", context=context)

    # 5) business_validator
    results["business_validator"] = call_node("business_validator", context=context)

    # 6) reporter
    results["reporter"] = call_node("reporter", context=context)


def main(run_once: bool = True, schedule_interval: Optional[int] = None) -> None:
    """
    run_once: if True run the workflow once and exit.
    schedule_interval: if provided (seconds) run repeatedly with sleep intervals.
    """
    if run_once:
        print(f"[workflow] Running once (SPEC_PATH={SPEC_PATH})")
        out = build_and_run_stategraph_once()
        try:
            print(json.dumps(out, indent=2, default=str))
        except Exception:
            print(out)
        return

    # scheduled loop
    if schedule_interval is None or schedule_interval <= 0:
        schedule_interval = 60
    print(f"[workflow] Starting scheduled runs every {schedule_interval} seconds (Ctrl-C to stop).")
    try:
        while True:
            build_and_run_stategraph_once()
            time.sleep(schedule_interval)
    except KeyboardInterrupt:
        print("[workflow] Scheduler stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Invoice Auditor workflow (modern LangGraph wrapper + fallback).")
    parser.add_argument("--run-once", action="store_true", help="Run the workflow once then exit (default behavior).")
    parser.add_argument("--schedule", type=int, default=0, help="If provided (>0), run repeatedly every N seconds.")
    args = parser.parse_args()

    if args.schedule and args.schedule > 0:
        main(run_once=False, schedule_interval=args.schedule)
    else:
        main(run_once=True)
