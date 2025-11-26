import json
from pathlib import Path
import langdetect
from agents.litellm_client import LiteLLMClient
PROCESSED = Path("data/processed")
client = LiteLLMClient()
def run(params, context):
    inp = PROCESSED / 'extractor_output.json'
    if not inp.exists():
        return {"status":"no_input"}
    data = json.load(open(inp))
    outputs = []
    for p in data.get('extracted_files', []):
        doc = json.load(open(p))
        txt = doc['raw_text']
        try:
            src = langdetect.detect(txt)
        except:
            src = 'en'
        doc['source_lang_detected'] = src
        if src != 'en':
            res = client.translate(txt, source_lang=src, target_lang='en')
            doc['translated_text'] = res.get('translated_text')
            doc['translation_confidence'] = res.get('confidence', 0.5)
            doc['translation_meta'] = {"provider": client.provider}
        else:
            doc['translated_text'] = txt
            doc['translation_confidence'] = 1.0
            doc['translation_meta'] = {"provider": "none"}
            # LLM-assisted structured extraction to refine the simple parser
        try:
            refined = client.extract_structured(doc['translated_text'])
            doc['structured_refined'] = refined
        except Exception as e:
            doc['structured_refined'] = doc.get('structured', {})
        outp = Path(p).with_suffix('.translated.json')
        json.dump(doc, open(outp, 'w'), indent=2)
        outputs.append(str(outp))
    out_path = PROCESSED / 'translation_output.json'
    json.dump({"translated_files": outputs}, open(out_path, 'w'), indent=2)
    return {"translated": outputs}

if __name__ == '__main__':
    print(run({},{}))