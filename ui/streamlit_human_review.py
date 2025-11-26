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

# Paths
BASE = Path.cwd()
REPORTS_DIR = BASE / "data" / "reports"
FEEDBACK_DIR = BASE / "data" / "feedback"
AUDIT_LOG = BASE / "data" / "audit_log.json"
SPEC_URL = "/mnt/data/AI Invoice Auditor Agentic AI.docx"  # uploaded spec path (per instruction)

# Ensure folders exist
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(AUDIT_LOG.parent).mkdir(parents=True, exist_ok=True)
if not AUDIT_LOG.exists():
    AUDIT_LOG.write_text("[]", encoding="utf-8")

st.set_page_config(page_title="Invoice Auditor — Human Review", layout="wide")

st.title("Invoice Auditor — Human-in-the-Loop Dashboard")
st.markdown(f"**Spec file**: `{SPEC_URL}` — (local path; your system will transform this to a usable URL).")

# Helper utilities
def load_report_paths():
    return sorted(list(REPORTS_DIR.glob("*.report.json")), key=lambda p: p.name)

def load_json(path):
    return json.loads(open(path, "r", encoding="utf-8").read())

def save_json(path, obj):
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

        # Lines editor (use pandas DataFrame + experimental_data_editor)
        st.subheader("Line items (editable)")
        lines = invoice.get("lines", [])
        if not lines:
            df = pd.DataFrame([{"item_code": "", "description":"", "qty": 0, "unit_price": 0.0, "total": 0.0}])
        else:
            df = pd.DataFrame(lines)
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
                        biz_issues = biz_mod.check_with_erp(struct_candidate)
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

