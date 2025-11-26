import json
from pathlib import Path
import yaml
import requests
from agents.litellm_client import LiteLLMClient
PROCESSED = Path('data/processed')
CONFIG = Path('configs/rules.yaml')
RULES = yaml.safe_load(open(CONFIG))
ERP_BASE = 'http://localhost:8000'
client = LiteLLMClient()
def validate_totals(struct):
    probs = []
    try:
        total = float(struct.get('total') or 0)
    except:
        probs.append({'code':'invalid_total','message':'Total missing or invalid','severity':'HIGH'})
        return probs
    sum_lines = 0
    for l in struct.get('lines', []):
        sum_lines += float(l.get('total',0))
    tol = RULES.get('tolerances',{}).get('total_tolerance_pct',1.0)
    if total>0 and abs(sum_lines - total)/ (total + 1e-9) * 100 > tol:
        probs.append({'code':'total_mismatch','message':f'Line items sum {sum_lines} differs from total {total} by >{tol}%', 'severity':'MED'})
    return probs
def run(params, context):
    inp = PROCESSED / 'translation_output.json'
    if not inp.exists():
        return {"status":"no_input"}
    data = json.load(open(inp))
    results = []
    for p in data.get('translated_files', []):
        doc = json.load(open(p))
        # prefer LLM-refined structured if present
        struct = doc.get('structured_refined') or doc.get('structured') or {}
        problems = []
        # required fields
        for f in RULES.get('required_fields',[]):
            if not struct.get(f):
                problems.append({'code':'missing_field','field':f,'severity':'MED','message':f'{f}is missing'})
        problems.extend(validate_totals(struct))
        # call LLM to evaluate and provide explanations
        try:
            eval_res = client.evaluate_invoice(struct)
            # merge LLM issues into problems (avoid duplication)
            for i in eval_res.get('issues', []):
                problems.append(i)
            llm_suggestion = eval_res.get('suggestion')
        except Exception as e:
            llm_suggestion = None
        outp = Path(p).with_suffix('.validated.json')
        json.dump({"structured": struct, "problems": problems, "llm_suggestion": llm_suggestion}, open(outp,'w'), indent=2)
        results.append(str(outp))
    out_path = PROCESSED / 'validator_output.json'
    json.dump({"validated_files": results}, open(out_path,'w'), indent=2)
    return {"validated": results}

if __name__ == '__main__':
    print(run({},{}))