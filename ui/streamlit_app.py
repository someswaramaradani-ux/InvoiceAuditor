# ui/streamlit_app.py
import streamlit as st
import json
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"

st.title("AI Invoice Auditor — Human Review")
reports = list(REPORT_DIR.glob("*.report.json"))
sel = st.selectbox("Select report", [r.name for r in reports] or ["No reports"])
if sel and sel != "No reports":
    data = json.load(open(REPORT_DIR / sel))
    st.header(f"Invoice: {data['invoice'].get('invoice_number')}")
    st.json(data)
    if st.button("Mark as Approved"):
        st.success("Marked Approved (store feedback)")
    if st.button("Request Manual Review"):
        st.warning("Marked for Manual Review")
