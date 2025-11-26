# agents/reporting_node.py  (patched run function)
import json
from pathlib import Path
from jinja2 import Template

PROCESSED = Path('data/processed')
REPORTS = Path('data/reports')
REPORTS.mkdir(parents=True, exist_ok=True)

TEMPLATE = '''
<html>
<head><title>Invoice Report - {{ invoice.invoice_number }}</title></head>
<body>
<h1>Invoice {{ invoice.invoice_number }}</h1>
<p>Vendor: {{ invoice.vendor }}</p>
<p>Total: {{ invoice.total }} {{ invoice.currency }}</p>
<h2>Problems</h2>
<ul>{% for p in problems %}<li>{{ p }}</li>{% else %}<li>None</li>{% endfor %}</ul>
</body>
</html>
'''

def safe_str(x, default=""):
    """Return a sensible string for values that might be None."""
    if x is None:
        return default
    return str(x)

def run(params, context):
    inp = PROCESSED / 'business_validator_output.json'
    if not inp.exists():
        return {"status":"no_input"}
    data = json.load(open(inp))
    outs = []
    for p in data.get('biz_validated', []):
        try:
            doc = json.load(open(p))
        except Exception:
            # malformed JSON or missing file, skip and continue
            continue

        struct = doc.get('structured') or {}
        # Coerce invoice id to a safe string; handles None value explicitly present
        invoice_id = safe_str(struct.get('invoice_number')) or "unknown"

        # Ensure we have safe values for template rendering
        safe_invoice = {
            "invoice_number": invoice_id,
            "vendor": safe_str(struct.get('vendor')),
            "total": safe_str(struct.get('total')),
            "currency": safe_str(struct.get('currency'))
        }

        problems = doc.get('problems') or []
        business_issues = doc.get('business_issues') or []
        # combine problems safely into list of strings or dicts
        combined_problems = []
        for item in problems + business_issues:
            combined_problems.append(item)

        # Write report JSON
        report_json_path = REPORTS / f"{invoice_id}.report.json"
        report_payload = {"invoice": safe_invoice, "problems": combined_problems}
        try:
            with open(report_json_path, 'w', encoding='utf-8') as fj:
                json.dump(report_payload, fj, indent=2, ensure_ascii=False)
        except Exception as e:
            # if write fails, record the error and continue
            outs.append({"error": f"failed_write_json_for_{invoice_id}", "detail": str(e)})
            continue

        # Render HTML report
        try:
            html = Template(TEMPLATE).render(invoice=safe_invoice, problems=combined_problems)
            html_path = REPORTS / f"{invoice_id}.report.html"
            with open(html_path, 'w', encoding='utf-8') as fh:
                fh.write(html)
        except Exception as e:
            outs.append({"error": f"failed_write_html_for_{invoice_id}", "detail": str(e)})
            # still include JSON path if it exists
            outs.append({"json": str(report_json_path)})
            continue

        outs.append({"json": str(report_json_path), "html": str(html_path)})
    out_path = PROCESSED / 'reporter_output.json'
    json.dump({"reports": outs}, open(out_path,'w'), indent=2)
    return {"reports": outs}

if __name__ == '__main__':
    print(run({},{}))
