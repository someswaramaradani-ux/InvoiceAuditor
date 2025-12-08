"""
Streamlit Human-in-the-Loop (HITL) Dashboard for Invoice Auditor

Place in: ui/streamlit_human_review.py
Run:    streamlit run ui/streamlit_human_review.py

This dashboard expects the repo layout described in the canvas docs:
 - data/reports/*.report.json
 - data/processed/* (optional)
 - agents/business_validation_node.py (optional, used for ERP cross-check)
 - agents/rag_agents/indexing_node.py (optional, used for reindex)
 - agents/litellm_client.py (optional, for LLM suggestions)

Feedback is saved to: data/feedback/<invoice>.feedback.json
Audit trail is appended to: data/audit_log.json
"""

import streamlit as st
from pathlib import Path
import json
import datetime
import pandas as pd
import importlib
import traceback
import sys
import os
import time
from typing import Optional, Any, Dict, List
import litellm

# Paths
BASE = Path.cwd()
REPORTS_DIR = BASE / "data" / "reports"
FEEDBACK_DIR = BASE / "data" / "feedback"
AUDIT_LOG = BASE / "data" / "audit_log.json"

# Ensure folders exist
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(AUDIT_LOG.parent).mkdir(parents=True, exist_ok=True)
if not AUDIT_LOG.exists():
    AUDIT_LOG.write_text("[]", encoding="utf-8")

st.set_page_config(page_title="Invoice Auditor — Dashboard", layout="wide")

st.title("Invoice Auditor — Dashboard")

# Helper utilities
def load_report_paths():
    return sorted(list(REPORTS_DIR.glob("*.report.json")), key=lambda p: p.name)

def load_json(path: Path):
    return json.loads(open(path, "r", encoding="utf-8").read())

def save_json(path: Path, obj):
    open(path, "w", encoding="utf-8").write(json.dumps(obj, indent=2, ensure_ascii=False))

def append_audit(entry):
    try:
        log = load_json(AUDIT_LOG)
    except Exception:
        log = []
    log.append(entry)
    save_json(AUDIT_LOG, log)

def human_username():
    # Minimal: could be replaced with real auth integration
    return st.session_state.get("reviewer", "human_reviewer")

# Sidebar: reviewer name and global controls
with st.sidebar:
    st.header("Reviewer & Controls")
    if "reviewer" not in st.session_state:
        st.session_state['reviewer'] = "alice@example.com"
    st.session_state['reviewer'] = st.text_input("Reviewer (email/name)", value=st.session_state['reviewer'])
    st.markdown("---")
    st.write("Actions")
    run_reindex = st.button("Re-run FAISS Index (indexer.run)", key="reindex")
    st.markdown("⚠️ Re-index will call `agents/rag_agents/indexing_node.run()` if present in repo.")
    st.markdown("---")
    st.write("Audit trail")
    if st.button("Show audit log"):
        try:
            st.json(load_json(AUDIT_LOG))
        except Exception as e:
            st.error("Could not read audit log: " + str(e))

# Left column: report list
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Reports")
    report_paths = load_report_paths()
    if not report_paths:
        st.info("No reports found in `data/reports/`. Run the pipeline or place a sample report file there.")
    selected = None
    names = [p.name for p in report_paths]
    selected_name = st.selectbox("Select report", ["-- none --"] + names)
    if selected_name != "-- none --":
        selected = REPORTS_DIR / selected_name

    st.markdown("---")
    st.write("Feedback files")
    fb_list = sorted(list(FEEDBACK_DIR.glob("*.feedback.json")))
    st.write([p.name for p in fb_list])
    if st.button("Show feedback folder"):
        st.write(fb_list)

# Right column: report viewer + editor + actions
with col2:
    if not selected:
        st.info("Select a report to view and review.")
    else:
        st.header(f"Review: {selected.name}")
        try:
            report = load_json(selected)
        except Exception as e:
            st.error("Failed to load report JSON: " + str(e))
            st.stop()

        invoice = report.get("invoice", {})
        problems = report.get("problems", [])
        llm_suggestion = report.get("llm_suggestion", None)

        # show original report
        with st.expander("Original report (JSON)", expanded=False):
            st.json(report)

        # Editable fields
        st.subheader("Invoice fields (editable)")
        c_invoice_number, c_date, c_vendor = st.columns([2,2,3])
        with c_invoice_number:
            invoice_number = st.text_input("Invoice Number", value=invoice.get("invoice_number", ""))
        with c_date:
            date_val = st.text_input("Date", value=invoice.get("date", ""))
        with c_vendor:
            vendor = st.text_input("Vendor", value=invoice.get("vendor", ""))

        c_total, c_currency, _ = st.columns([2,1,1])
        with c_total:
            total_val = st.text_input("Total", value=str(invoice.get("total", "")))
        with c_currency:
            currency = st.text_input("Currency", value=invoice.get("currency", ""))

        # Lines editor (use pandas DataFrame + data_editor)
        st.subheader("Line items (editable)")
        lines = invoice.get("lines", [])
        if not lines:
            df = pd.DataFrame([{"item_code": "", "description":"", "qty": 0, "unit_price": 0.0, "total": 0.0}])
        else:
            df = pd.DataFrame(lines)
        # Use st.data_editor (compatibility ensured earlier in conversation)
        edited_df = st.data_editor(
                    df,
                    num_rows="dynamic",
                    width='stretch'
                )

        # Comments / reviewer notes
        st.subheader("Reviewer notes")
        note = st.text_area("Add a comment / reasoning for decision", value="", height=120)

        # Show problems computed by pipeline
        st.subheader("Pipeline Problems")
        st.write("Reported problems:")
        if problems:
            for p in problems:
                st.write("- ", p)
        else:
            st.write("No problems reported.")

        if llm_suggestion:
            st.info(f"LLM suggestion: {llm_suggestion}")

        # Buttons for actions
        cola, colb, colc, cold = st.columns(4)
        with cola:
            if st.button("Approve", key="approve"):
                status = "APPROVED"
                # create feedback object and save
                feedback = {
                    "invoice_id": invoice_number,
                    "status": status,
                    "reviewer": human_username(),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "note": note,
                    "edited_invoice": {
                        "invoice_number": invoice_number,
                        "date": date_val,
                        "vendor": vendor,
                        "currency": currency,
                        "total": total_val,
                        "lines": edited_df.fillna("").to_dict(orient="records")
                    },
                    "original_report": str(selected),
                }
                fb_path = FEEDBACK_DIR / f"{invoice_number}.feedback.json"
                save_json(fb_path, feedback)
                append_audit({"invoice_id": invoice_number, "action": "approve", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "feedback_path": str(fb_path)})
                st.success(f"Invoice {invoice_number} marked APPROVED. Feedback saved to {fb_path}")

        with colb:
            if st.button("Manual Review", key="manual"):
                status = "MANUAL_REVIEW"
                feedback = {
                    "invoice_id": invoice_number,
                    "status": status,
                    "reviewer": human_username(),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "note": note,
                    "edited_invoice": {
                        "invoice_number": invoice_number,
                        "date": date_val,
                        "vendor": vendor,
                        "currency": currency,
                        "total": total_val,
                        "lines": edited_df.fillna("").to_dict(orient="records")
                    },
                    "original_report": str(selected),
                }
                fb_path = FEEDBACK_DIR / f"{invoice_number}.feedback.json"
                save_json(fb_path, feedback)
                append_audit({"invoice_id": invoice_number, "action": "manual_review", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "feedback_path": str(fb_path)})
                st.warning(f"Invoice {invoice_number} marked for MANUAL REVIEW. Feedback saved to {fb_path}")

        with colc:
            if st.button("Reject", key="reject"):
                status = "REJECTED"
                feedback = {
                    "invoice_id": invoice_number,
                    "status": status,
                    "reviewer": human_username(),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "note": note,
                    "edited_invoice": {
                        "invoice_number": invoice_number,
                        "date": date_val,
                        "vendor": vendor,
                        "currency": currency,
                        "total": total_val,
                        "lines": edited_df.fillna("").to_dict(orient="records")
                    },
                    "original_report": str(selected),
                }
                fb_path = FEEDBACK_DIR / f"{invoice_number}.feedback.json"
                save_json(fb_path, feedback)
                append_audit({"invoice_id": invoice_number, "action": "reject", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "feedback_path": str(fb_path)})
                st.error(f"Invoice {invoice_number} marked REJECTED. Feedback saved to {fb_path}")

        with cold:
            if st.button("Re-run validation (local)"):
                # run local validation similar to validation_node
                try:
                    # Basic totals validation
                    try:
                        tot = float(total_val)
                    except:
                        st.error("Total is not a valid number.")
                        tot = None
                    lines_list = edited_df.fillna("").to_dict(orient="records")
                    lines_sum = 0.0
                    for L in lines_list:
                        try:
                            lines_sum += float(L.get("total", 0) or 0)
                        except:
                            pass
                    problems_local = []
                    if tot is not None and abs(lines_sum - tot) > 0.01:
                        problems_local.append({"code":"total_mismatch", "message": f"Line sum {lines_sum} != total {tot}"})
                    # business validation via agents.business_validation_node.check_with_erp if available
                    try:
                        biz_mod = importlib.import_module("agents.business_validation_node")
                        # prepare struct in the same shape expected
                        struct_candidate = {
                            "invoice_number": invoice_number,
                            "date": date_val,
                            "vendor": vendor,
                            "currency": currency,
                            "total": total_val,
                            "lines": lines_list
                        }
                        # try both styles: check_with_erp exists or evaluate_invoice
                        if hasattr(biz_mod, "run"):
                            # run returns manifest; not ideal — skip
                            biz_issues = biz_mod.run()
                        else:
                            biz_issues = ["business_validation_not_available"]
                    except Exception as e:
                        biz_issues = ["business_validation_not_available"]
                    st.write("Local validation results:")
                    st.json({"totals_check": problems_local, "business_issues": biz_issues})
                    append_audit({"invoice_id": invoice_number, "action":"re_run_validation", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "result": {"totals_check": problems_local, "business_issues": biz_issues}})
                except Exception as e:
                    st.error("Error during re-run validation: " + str(e))
                    st.exception(traceback.format_exc())

        # Optionally re-index
        if run_reindex:
            try:
                idx_mod = importlib.import_module("agents.rag_agents.indexing_node")
                st.info("Running indexer...")
                idx_res = idx_mod.run({}, {})
                st.success(f"Indexer completed: {idx_res}")
                append_audit({"invoice_id": invoice_number, "action":"reindex", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "result": idx_res})
            except Exception as e:
                st.error("Failed to run indexer: " + str(e))

        # Allow downloading edited invoice (as JSON)
        if st.button("Save edited invoice into report (overwrite)"):
            # Overwrite current report with edited invoice content and updated problems/metadata
            new_report = {
                "invoice": {
                    "invoice_number": invoice_number,
                    "date": date_val,
                    "vendor": vendor,
                    "currency": currency,
                    "total": total_val,
                    "lines": edited_df.fillna("").to_dict(orient="records")
                },
                "problems": problems,
                "llm_suggestion": llm_suggestion
            }
            dst = REPORTS_DIR / f"{invoice_number}.report.json"
            save_json(dst, new_report)
            append_audit({"invoice_id": invoice_number, "action":"overwrite_report", "reviewer": human_username(), "time": datetime.datetime.utcnow().isoformat(), "report_path": str(dst)})
            st.success(f"Report overwritten and saved to {dst}")

        # Show a preview HTML (if exists)
        html_path = REPORTS_DIR / f"{invoice.get('invoice_number','unknown')}.report.html"
        if html_path.exists():
            st.markdown("---")
            st.subheader("HTML preview")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=400, scrolling=True)


# ----------------------------
# RAG Search integration (appended panel)
# ----------------------------
# The RAG panel uses the retrieval_node.run() if present, otherwise falls back to
# FAISS + sentence-transformers local search. It also uses LiteLLMClient (if installed)
# to generate grounded answers and records HITL feedback to data/hitl/feedback.jsonl.

st.markdown("---")
st.header("RAG Search (Invoice Index)")
st.markdown("Search the indexed invoice reports and provide HITL feedback or generate grounded answers.")

# Ensure HITL feedback dir
HITL_DIR = BASE / "data" / "hitl"
HITL_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_PATH = HITL_DIR / "feedback.jsonl"

# Try to import retrieval node
RETRIEVAL_AVAILABLE = False
retrieval_mod = None
try:
    retrieval_mod = importlib.import_module("agents.rag_agents.retrieval_node")
    RETRIEVAL_AVAILABLE = hasattr(retrieval_mod, "run")
except Exception:
    RETRIEVAL_AVAILABLE = False

# Try to import LiteLLM client
LLM_AVAILABLE = False
from agents.litellm_client import LiteLLMClient
client = LiteLLMClient()
try:
    lite_mod = importlib.import_module("agents.litellm_client")
    LiteLLMClient = getattr(lite_mod, "LiteLLMClient", None)
    if LiteLLMClient is not None:
        LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# Local FAISS fallback availability
LOCAL_FAISS_AVAILABLE = False
try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore
    LOCAL_FAISS_AVAILABLE = True
except Exception:
    LOCAL_FAISS_AVAILABLE = False

# UI controls
cols = st.columns([4,1])
with cols[0]:
    query = st.text_input("RAG Query", value="Which invoices have total mismatches?")
with cols[1]:
    top_k = st.number_input("Top K", min_value=1, max_value=10, value=3)
use_node = st.checkbox("Use retrieval node (if available)", value=True)
if LLM_AVAILABLE:
    provider_choice = st.selectbox("LiteLLM provider (env override)", ["mock","local","hf_api"])
    # set env override if selected
    os.environ["LITELLM_PROVIDER"] = provider_choice

if st.button("Search RAG"):
    if not query.strip():
        st.warning("Please provide a query.")
    else:
        # Retrieval wrapper
        def _load_report(path: Path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {"_raw": path.read_text(encoding="utf-8")}

        def _local_faiss_search(q: str, k: int = 3):
            idx_file = BASE / "data" / "faiss" / "invoices.index"
            meta_file = BASE / "data" / "faiss" / "meta.json"
            if not idx_file.exists() or not meta_file.exists() or not LOCAL_FAISS_AVAILABLE:
                st.warning("Local FAISS index or sentence-transformers not available. Build index first.")
                return []
            import numpy as np
            index = faiss.read_index(str(idx_file))
            model = SentenceTransformer("all-MiniLM-L6-v2")
            qemb = model.encode([q])
            D, I = index.search(np.array(qemb).astype("float32"), k)
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            results = []
            for i in I[0]:
                if i < 0 or i >= len(meta):
                    continue
                report_path = Path(meta[i])
                report_json = _load_report(report_path)
                results.append({"score": 0.0, "path": str(report_path), "report": report_json})
            return results

        def _retrieve(q: str, k: int = 3):
            if use_node and RETRIEVAL_AVAILABLE and retrieval_mod is not None:
                try:
                    out = retrieval_mod.run({"query": q, "k": k}, {})
                    # normalize many shapes
                    if isinstance(out, dict) and "results" in out:
                        normalized = []
                        for r in out["results"][:k]:
                            if isinstance(r, dict) and r.get("path"):
                                report_json = _load_report(Path(r["path"]))
                                normalized.append({"score": r.get("score", 0.0), "path": r.get("path"), "report": report_json})
                            else:
                                normalized.append({"score": 0.0, "path": None, "report": r})
                        return normalized
                    if isinstance(out, list):
                        return [{"score": 0.0, "path": None, "report": r} for r in out][:k]
                    if isinstance(out, dict):
                        return [{"score": 0.0, "path": None, "report": out}][:k]
                except Exception as e:
                    st.warning(f"retrieval_node.run failed: {e}")
            return _local_faiss_search(q, k)

        # run retrieval
        with st.spinner("Retrieving..."):
            results = _retrieve(query, top_k)

        if not results:
            st.info("No results found.")
        else:
            for idx, r in enumerate(results):
                st.markdown("---")
                left, right = st.columns([4,1])
                path = r.get("path")
                report = r.get("report") or {}
                # Provide a compact summary/snippet
                snippet = ""
                try:
                    invoice = report.get("invoice") or report.get("original_validated_doc", {}).get("structured") or report
                    inv_num = (invoice.get("invoice_number") if isinstance(invoice, dict) else None) or "unknown"
                    vendor = invoice.get("vendor") if isinstance(invoice, dict) else None
                    total = invoice.get("total") if isinstance(invoice, dict) else None
                    snippet = f"Invoice: **{inv_num}**  | Vendor: **{vendor}**  | Total: **{total}**"
                except Exception:
                    snippet = str(report)[:200]
                with left:
                    st.markdown(f"**Result #{idx+1}**  \n{snippet}")
                    st.expander("Report JSON", expanded=False).json(report)
                with right:
                    if st.button(f"Mark Relevant {idx}", key=f"rag_rel_{idx}"):
                        entry = {"timestamp": int(time.time()), "query": query, "item_path": path, "label": "relevant", "score": r.get("score",0.0)}
                        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        st.success("Marked relevant ✅")
                    if st.button(f"Mark Irrelevant {idx}", key=f"rag_irr_{idx}"):
                        entry = {"timestamp": int(time.time()), "query": query, "item_path": path, "label": "irrelevant", "score": r.get("score",0.0)}
                        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        st.warning("Marked irrelevant ❌")
                    if path:
                        st.write(path)

            # Generate grounded answer
            st.markdown("---")
            if LLM_AVAILABLE:
                if st.button("Generate grounded answer (LiteLLM)"):
                    ctx_parts = []
                    for r in results:
                        report = r.get("report", {})
                        ctx_parts.append(json.dumps(report, ensure_ascii=False)[:4000])
                    context_text = "\n\n---\n\n".join(ctx_parts)
                    try:
                        client = LiteLLMClient()
                        system = "You are a concise assistant. Use the provided documents to answer the user's query. If unsure, say 'I don't know.' Return plain text and cite sources when possible. Return your answer in pure JSON format."
                        user = f"User query: {query}\n\nContext:\n{context_text}"
                        gen = client.generate(system, user, max_tokens=512, temperature=0.0)
                        messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user}
                        ]
                        try:
                            response = litellm.completion(
                            model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
                            messages=messages,
                            api_base="http://localhost:11434",
                            temperature=0.0  # Set low temperature for literal, non-creative translation
                            )
                            extracted_text = response['choices'][0]['message']['content'].strip()
                            print(f"extracted_text::: {extracted_text}\n")
                            #parsed = json.loads(extracted_text)
                            #print(f"parsed: {parsed}")
                            #if isinstance(parsed, dict):
                                #text = gen.get("text") if isinstance(gen, dict) else str(gen)
                            st.subheader("Generated Answer")
                            st.write(extracted_text)
                        except Exception as e:
                            st.error(f"LLM generate failed: {e}")
                    except Exception as e:
                        st.error(f"LLM generate failed: {e}")
            else:
                st.info("LiteLLM client not available; install or set LITELLM_PROVIDER.")

# Show recent feedback (sidebar)
with st.sidebar:
    st.markdown("---")
    st.markdown("### Recent RAG feedback")
    try:
        if FEEDBACK_PATH.exists():
            lines = FEEDBACK_PATH.read_text(encoding="utf-8").strip().splitlines()[-10:]
            for ln in reversed(lines):
                try:
                    st.write(json.loads(ln))
                except Exception:
                    st.text(ln)
        else:
            st.text("No feedback yet.")
    except Exception:
        st.text("No feedback yet.")
