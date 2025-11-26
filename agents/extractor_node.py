import json
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import docx
import re

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for pg in pdf.pages:
            text += (pg.extract_text() or "") + "\n"
    return text


def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text_from_image(path):
    return pytesseract.image_to_string(Image.open(path))


def simple_parse_fields(text):
    d = {"invoice_number": None, "date": None, "vendor": None, "currency": None, "total": None, "lines": []}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines[:120]:
        if not d['invoice_number'] and re.search(r"invoice\s*(no|#|number)[:\s]*([A-za-z0-9-]+)", l, re.I):
            d['invoice_number'] = re.sub(r".*invoice\s*(no|#|number)[:\s]*", "", l, flags=re.I)
        if not d['total'] and re.search(r"total[:\s]*([\d,]+\.\d{2})", l, re.I):
            d['total'] = re.search(r"([\d,]+\.\d{2})", l).group(1).replace(',','')
        if not d['vendor'] and len(l.split())<=6 and any(k in l.lower() for k in ['ltd','inc','co.','company','srl','llc']):
            d['vendor'] = l
    return d


def run(params, context):
    monitor_out = PROCESSED / 'monitor_output.json'
    if not monitor_out.exists():
        return {"status":"no_files"}
    monitor = json.load(open(monitor_out))
    results = []
    for f in monitor.get('files', []):
        path = Path(f['path'])

        # Skip hidden files and macOS .DS_Store
        if path.name.startswith('.') or path.suffix.lower() not in ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt']:
            print(f"Skipping unsupported or hidden file: {path.name}")
            continue

        text = ""
        if path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(path)
        elif path.suffix.lower() == '.docx':
            text = extract_text_from_docx(path)
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text = extract_text_from_image(path)
        elif path.suffix.lower() == '.txt':
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            print(f"Skipping unrecognized file type: {path}")
            continue

        structured = simple_parse_fields(text)
        out = {"source": str(path), "raw_text": text, "structured": structured}
        out_file = PROCESSED / (path.stem + '.extracted.json')
        with open(out_file, 'w') as f:
            json.dump(out, f, indent=2)
        results.append(str(out_file))
    out_path = PROCESSED / 'extractor_output.json'
    json.dump({"extracted_files": results}, open(out_path, 'w'), indent=2)
    return {"extracted": results}

if __name__ == '__main__':
    print(run({},{}))