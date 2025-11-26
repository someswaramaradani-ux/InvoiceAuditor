# mock_erp/app.py
from fastapi import FastAPI, HTTPException
import json
import os

app = FastAPI()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PO_FILE = os.path.join(DATA_DIR, "po_records.json")
VENDOR_FILE = os.path.join(DATA_DIR, "vendors.json")

with open(PO_FILE) as f:
    PO_RECORDS = json.load(f)

with open(VENDOR_FILE) as f:
    VENDORS = json.load(f)

@app.get("/po/{po_number}")
def get_po(po_number: str):
    rec = PO_RECORDS.get(po_number)
    if not rec:
        raise HTTPException(status_code=404, detail="PO not found")
    return rec

@app.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    rec = VENDORS.get(vendor_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return rec
