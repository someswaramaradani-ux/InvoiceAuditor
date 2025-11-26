import json
from pathlib import Path
import requests
PROCESSED = Path('data/processed')
ERP_BASE = 'http://localhost:8000'
def check_with_erp(struct):
    issues = []
    po = struct.get('po_number')
    if not po:
        issues.append('po_missing')
        return issues
    r = requests.get(f"{ERP_BASE}/po/{po}")
    if r.status_code != 200:
        issues.append('po_not_found')
        return issues
    po_rec = r.json()
    tol = 5.0
    for line in struct.get('lines',[]):
        code = line.get('item_code')
        erp_line = next((l for l in po_rec.get('lines',[]) if
                 l.get('item_code')==code), None)
        if not erp_line:
            issues.append(f'item_{code}_not_in_po')
            continue
        if abs(float(line.get('unit_price',0)) - float(erp_line.get('unit_price',0))) / (float(erp_line.get('unit_price',1)))* 100 > tol:
            issues.append(f'price_mismatch_{code}')
    return issues
def run(params, context):
    inp = PROCESSED / 'validator_output.json'
    if not inp.exists():
        return {"status":"no_input"}
    data = json.load(open(inp))
    outs = []
    for p in data.get('validated_files',[]):
        doc = json.load(open(p))
        struct = doc.get('structured', {})
        business_issues = check_with_erp(struct)
        doc['business_issues'] = business_issues
        outp = Path(p).with_suffix('.bizvalidated.json')
        json.dump(doc, open(outp,'w'), indent=2)
        outs.append(str(outp))
    out_path = PROCESSED / 'business_validator_output.json'
    json.dump({"biz_validated": outs}, open(out_path,'w'), indent=2)
    return {"biz_validated": outs}

if __name__ == '__main__':
    print(run({},{}))