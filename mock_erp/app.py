# mock_erp/app.py
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import json
import os

app = FastAPI()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PO_FILE = os.path.join(DATA_DIR, "PO_records.json")
VENDOR_FILE = os.path.join(DATA_DIR, "vendors.json")

with open(PO_FILE) as f:
    po_list = json.load(f)

with open(VENDOR_FILE) as f:
    vendors_list = json.load(f)

@app.get("/po/{po_number}")
def get_po(po_number: str):
    PO_RECORDS: Dict[str, Any] = {
        po["po_number"]: po for po in po_list if "po_number" in po
    }
    rec = PO_RECORDS.get(po_number)
    if not rec:
        raise HTTPException(status_code=404, detail="PO not found")
    return rec

@app.get("/vendor/{vendor_id}")
def get_vendor(vendor_id: str):
    VENDORS: Dict[str, Any] = {
        vl["vendor_id"]: vl for vl in vendors_list if "vendor_id" in vl
    }
    rec = VENDORS.get(vendor_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return rec
