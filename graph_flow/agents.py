# NOTE: These are simplified stubs to build the graph structure.
from typing import Dict,Any
import os
import json
import logging
from state import InvoiceProcessingState
from logger_config import AUDITOR_LOGGER
import json
from pathlib import Path
import pdfplumber
import pytesseract
from PIL import Image
import docx
import re
import re
from typing import Dict, List
from logger_config import AUDITOR_LOGGER
from state import InvoiceProcessingState 
from schemas import InvoiceExtractionSchema, LineItem, ValidationReport, Issue, ValidationResults
from langdetect import detect
import litellm

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

def ingest_invoice(state: InvoiceProcessingState) -> Dict:
    #Logic: Monitor /data/incoming, move file, update path and status.
    AUDITOR_LOGGER.info(f"Node: ingest_invoice | Starting ingestion for {state['file_name']}")
    #print(f"--- 1. Ingesting: {state['file_name']}")
    return {"status": "PROCESSING", "raw_text": "Simulated raw text from invoice."}

def extract_data(state: InvoiceProcessingState) -> Dict:
    AUDITOR_LOGGER.info(f"Node: extract_data | Starting extraction for {state['file_path']}")
    file_path = state['file_path']
    extracted_data = {}
    try:
        path = Path(file_path)

        # Skip hidden files and macOS .DS_Store
        if path.name.startswith('.') or path.suffix.lower() not in ['.pdf', '.docx', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.txt']:
            AUDITOR_LOGGER.info(f"Skipping unsupported or hidden file: {path.name}")

        text = ""
        if path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(path)
        elif path.suffix.lower() == '.docx':
            text = extract_text_from_docx(path)
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text = extract_text_from_image(path)
        elif path.suffix.lower() == '.txt':
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            AUDITOR_LOGGER.info(f"Skipping unrecognized file type:: {path}")
        AUDITOR_LOGGER.info(f"raw text: {text}")
        #extracted_data = simple_parse_fields(text)
        #AUDITOR_LOGGER.info(f"extracted_data:: {extracted_data}")
        try:
            # Detect the language using the first 500 characters for speed
            lang = detect(text[:500]).upper()
        except Exception as e:
            lang = 'UNKNOWN'
            AUDITOR_LOGGER.error(f"Language detection failed: {e}")
        #final_data = InvoiceExtractionSchema(**extracted_data) 
        #AUDITOR_LOGGER.info(f"Node: extract_data | Extracted Invoice No: {final_data.invoice_number}")

        return {
                "detected_language": lang,
                "extracted_data": text,
                "errors": state['errors']
        }
    except Exception as e:
        error_msg = f"File Extraction Failed: {e}"
        AUDITOR_LOGGER.error(error_msg)
        state['errors'].append(error_msg)
        # Ensure a minimal structure is returned to allow the workflow to proceed/fail gracefully
        return {"errors": state['errors'], "detected_language": lang, "extracted_data": {}}


'''def extract_data(state: InvoiceProcessingState) -> Dict:
    # Logic: OCR/Parsing/LLM extraction via LiteLLM/Azure AI.
    try:
        # --- Extraction Logic ---
        extracted_data = {"invoice_number": "INV-123", "total_amount": 1000.00}
        
        AUDITOR_LOGGER.info(f"Node: extract_data | Extraction successful. Invoice No: {extracted_data['invoice_number']}")
        return {"extracted_data": extracted_data}
        
    except Exception as e:
        AUDITOR_LOGGER.error(f"Node: extract_data | Failed during extraction: {e}")
        state['errors'].append(f"Extraction Error: {e}")
        return {"errors": state['errors']}'''

def translate_data(state: InvoiceProcessingState) -> Dict:
    AUDITOR_LOGGER.info(f"Node: translation_agent | Starting translation for language: {state.get('detected_language')}")
    
    # 1. Get necessary state variables
    extracted_data = state['extracted_data']
    source_lang = state['detected_language']
    
    # 3. Prepare LLM Prompt
    system_prompt = (
            "You are an expert, precise, and literal translator. "
            "Your task is to translate the user's text into **English**. "
            "**CRITICAL RULE:** You must strictly preserve all numerical values, "
            "including digits, currency symbols (e.g., $, €, ¥), percentage signs (%), "
            "and unit abbreviations (e.g., km, kg, lbs) exactly as they appear in the source text. "
            "Only translate the words and provide translation confidence in numericals(decimal format)."
            "return JSON: {tranlated_text, confidence}. Return pure valid JSON object only."
        )
    user_message = (
            f"Source Language: {source_lang}\n"
            f"Text to translate:\n"
            f"---START---\n"
            f"{extracted_data}\n"
            f"---END---"
            )

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
            ]
        
    try:
        # 4. Call LLM via LiteLLM (Using an accurate, fast Azure model)
        # Using a model like GPT-4o or GPT-4-Turbo is recommended for complex translations.
        response = litellm.completion(
        model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
        #model="ollama/llama3.1:latest",
        messages=messages,
        api_base="http://localhost:11434",
        temperature=0.0  # Set low temperature for literal, non-creative translation
        )
            
        res = response['choices'][0]['message']['content'].strip()
        AUDITOR_LOGGER.info(f"Node: translation_agent | Translation completed")
        #AUDITOR_LOGGER.info(f"Node: translation_agent | Response from translation agent: {res}")
        data = json.loads(res)
        #AUDITOR_LOGGER.info(f"Node: translation_agent | Response from translation agent: {res}")
        # 2. Access the values using their keys
        translated_text = data["translated_text"]
        translation_confidence = data["confidence"]
        AUDITOR_LOGGER.info(f"Node: translation_agent | Translation completed. Translated text: {translated_text}")
        AUDITOR_LOGGER.info(f"Node: translation_agent | Translation completed. Confidence: {translation_confidence}")
        structured_text = extract_structured(translated_text)
        AUDITOR_LOGGER.info(f"Node: structure text: {structured_text}")
        return {
                "translated_data": structured_text,
                "translation_confidence": translation_confidence,
                "errors": state['errors']
        }

    except Exception as e:
        error_msg = f"Translation LLM Error via LiteLLM: {e}"
        AUDITOR_LOGGER.error(error_msg)
        state['errors'].append(error_msg)
        return {
                "translated_data": {},
                "translation_confidence": 0.0,
                "errors": state['errors']
            }
        
   

def extract_structured(input_text: str) -> Dict[str, Any]:
        system_prompt = """You are an invoice extraction assistant. Extract invoice fields as JSON: 
        invoice_number, po_number, date, vendor, vendor_id, currency, total, taxes, line_items (list of {item_code, description, qty, unit_price, total}). 
        You neeed to fetch the value for currency as currency_name(currency_symbol)
        if unable to extract any of the fields, skip returning in json".
        Return pure JSON only."""
        user_prompt = (f"input json: {input_text}\n")
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
        
        AUDITOR_LOGGER.info(f"TranslationvNode:getting structure text")
        try:
            #litellm._turn_on_debug()
            response = litellm.completion( 
            model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
            #model="ollama/llama3.1:latest",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=0.0  # Set low temperature for literal, non-creative translation
            )
            res = response['choices'][0]['message']['content'].strip()
            #AUDITOR_LOGGER.info(f"Node: translated raw response: {res}")
            parsed = json.loads(res)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            # fallback heuristics minimal
            error_msg = f"Structured text extraction via LiteLLM: {e}"
            AUDITOR_LOGGER.error(error_msg)
            import re
            m = re.search(r"invoice\s*(no|number|#)[:\s]*([A-Za-z0-9-]+)", input_text, re.I)
            invoice_number = m.group(2) if m else None
            m2 = re.search(r"total[:\s]*([\d,]+\.\d{2})", input_text, re.I)
            total = m2.group(1).replace(",", "") if m2 else None
            return {"invoice_number": invoice_number, "date": None, "vendor": None, "currency": None, "total": total, "lines": []}
        
'''def translate_data(state: InvoiceProcessingState) -> Dict:
    # Logic: Language detection, LLM translation via LiteLLM.
    print("--- 3. Translating Data...")
    return {"translated_data": state['extracted_data'], "source_language": "German", "translation_confidence": 0.98}'''


# graph_flow/agents.py

import yaml
import os
from typing import Dict, Any, List
from pydantic import ValidationError


CONFIG_PATH = os.path.join(os.getcwd(), 'configs', 'rules.yaml')

def load_rules(path: str) -> Dict[str, Any]:
    """Reads the validation rules from the YAML file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load rules.yaml from {path}: {e}")
        return {}

def deterministic_validation(translated_data: Dict, rules: Dict) -> List[str]:
    """Performs non-LLM based (reliable) checks."""
    issues = []
    
    # 1. Check Allowed Currencies (from rules.yaml)
    currency = translated_data.get('currency', '').upper()
    allowed_currencies = [c.upper() for c in rules.get('allowed_currencies', [])]

    if currency and currency not in allowed_currencies:
        issues.append({'code':'CURRENCY_NOT_ALLOWED','field':'CURRENCY_NOT_ALLOWED','severity':'High','message':f"Currency '{currency}' is not in the allowed list: {', '.join(allowed_currencies)}'"})
    
    # 2. Verify Totals and Tax Calculations (without rules.yaml)
    try:
        subtotal = sum(item.get('line_amount', 0.0) for item in translated_data.get('line_items', []))
        total_amount = translated_data.get('total_amount', 0.0)
        
        # Simple check: (Line Item Sum + Tax) should equal Total Amount
        tax_amount = translated_data.get('tax_amount', 0.0)
        calculated_total = round(subtotal + tax_amount, 2)
        
        if abs(calculated_total - total_amount) > 0.01: # Use a tolerance for float math
            issues.append({'code':'CALC_MISMATCH','field':'total_amount, tax_amount','severity':'High','message': f"Calculation Error: Extracted Total ({total_amount}) does not match "
                         f"calculated total ({calculated_total} = Subtotal {round(subtotal, 2)} + Tax {round(tax_amount, 2)})."})
    
    except Exception as e:
        issues.append({'code':'CALC_PARSE_ERROR','field':'line_items, totals','severity':'High','message':f"Failed to parse numbers for calculation verification: {e}"})
    
    return issues

def validate_invoice_data(state: InvoiceProcessingState) -> Dict:
    """Performs validation checks using Python for math/files and LLM for complex logic."""
    AUDITOR_LOGGER.info("Node: validation_agent | Starting validation Agent:")
    rules = load_rules(CONFIG_PATH)
    rules_yaml_str = yaml.dump(rules)
    translated_data = state.get('translated_data', {})
    AUDITOR_LOGGER.info(f"Node: validation_agent | translated_data:{translated_data}")
    # --- STEP 1: Run Deterministic Checks (Python Logic) ---
    #python_issues = deterministic_validation(translated_data, rules)
    #AUDITOR_LOGGER.info(f"Node: validation_agent | python_issues:{python_issues}")
    # --- STEP 2: Use LLM for Completeness and Final JSON Formatting ---

    
    system_prompt = """
    You are an expert Invoice Validation Agent. Your task is to check the provided invoice data against the given rules and calculations.
    **RULES & INSTRUCTIONS:**
    1. **Completeness Check:** 
        - You will compare all fields listed in `required_fields` from the YAML rules file against invoice data. \n
        - If po_number, invoice_date, invoice_id are misssing in invoice data, mark as 'High' severity. \n
        - for currency field, extract based on currency used against amounts provided in the invoice.\n
        - for completeness check for fields in input data, exclude them if not mentioned in `required_fields` of yaml rules file.\n\n
    2. **Consistency Check:** Analyze the data fields for any non-compliance with the rules, and check the context for any obvious inconsistencies (e.g., date of invoice is in the future).\n
    3. **Severity:** Use 'High' for critical errors (calculation, forbidden currency), 'Medium' for required but missing fields, and 'Low' for minor inconsistencies.\n
    4. keep unicode characters as it is without any changes especially currency symbols.\n
    5. map currency symbols with currency_symbol_map in YAML rules and then validate against accepted_currencies in YAML rules.\n
    7. return JSON: {validation_results: {data_valid : True/False, completeness_ok : True/False},  issues: [{field, severity, message}] Return pure JSON only.\n
    8. Only output the JSON object. Do NOT include any explanatory text, markdown formatting.
    """
    
    user_prompt = (
         "**DATA CONTEXT:**\n"
        "1. **YAML Validation Rules:**\n"
        "{rules_yaml_str}\n"
        "2. **INVOICE DATA TO VALIDATE:**\n"
        "{translated_data}\n"
        "Analyze the data and produce the JSON response.\n"
        ).format(rules_yaml_str=rules_yaml_str, translated_data=translated_data)
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    
    try:
        response = litellm.completion(
            model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
            #model="ollama/llama3.1:latest",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=0.0
        )
        structured_text_str = response['choices'][0]['message']['content'].strip()
        AUDITOR_LOGGER.info(f"Node: validation agent| validation completed{structured_text_str}")
        data = json.loads(structured_text_str)
        AUDITOR_LOGGER.info(f"Node: validation_agent | validation completed. data: {data}")
        # 2. Access the values using their keys
        validation_results = data.get("validation_results", {}) if isinstance(data, dict) else {}
        
        validation_issues = []
        for i in data.get('issues', []):
                validation_issues.append(i)
        
        AUDITOR_LOGGER.info(f"Node: validation_agent | validation completed. validation results: {validation_results}")
        
        return {
            "validation_results": validation_results,
            "validation_issues": validation_issues,
            "validation_rules_yaml": rules_yaml_str
        }

    except Exception as e:
        error_msg = f"Validation LLM Error via LiteLLM: {e}"
        AUDITOR_LOGGER.error(error_msg)
        return {"validation_results": {},  "validation_issues": [], "validation_rules_yaml": rules_yaml_str, "errors": state['errors'] + [error_msg]}

'''def validate_invoice_data(state: InvoiceProcessingState) -> Dict:
    # Logic: Check against rules.yaml (completeness, calculations).
    print("--- 4. Validating Invoice Data (Rules Check)...")
    # Simulation: Assume some minor inconsistency
    return {"validation_results": {"data_valid": True, "completeness_ok": True, "calculation_error": False}}'''
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import requests

ERP_BASE = os.getenv("ERP_BASE", "http://127.0.0.1:8000")

def fetch_po(po_number: str) -> Optional[Dict[str, Any]]:
    """Fetch PO record from ERP. Returns dict or None on not found / error."""
    if requests is None:
        raise RuntimeError("requests package is required for business_validation_node (pip install requests)")
    try:
        resp = requests.get(f"{ERP_BASE}/po/{po_number}", timeout=100)
        if resp.status_code == 200:
            AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | https response: {resp}")
            return resp.json()
        else:
            return None
    except Exception as e:
        AUDITOR_LOGGER.error(f"Node: busiess_validation_agent | https response error: {e}")
        raise


def fetch_vendor(vendor_id: str) -> Optional[Dict[str, Any]]:
    """Fetch vendor info from ERP."""
    if requests is None:
        raise RuntimeError("requests package is required for business_validation_node (pip install requests)")
    try:
        resp = requests.get(f"{ERP_BASE}/vendor/{vendor_id}", timeout=100)
        if resp.status_code == 200:
            AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | https response: {resp}")
            return resp.json()
        else:
            return None
    except Exception as e:
        print("[demo] ERROR:", e)
        raise

def business_validation_llm_call(structured: Dict[str, Any], erp_json, validation_type: str) :
        #system = """You are a forensic invoice auditor. Evaluate this structured invoice JSON for likely errors and return JSON: 
        #{issues: [{code, field, severity, message}], suggestion: 'Approve|Manual Review|Reject'}. Return pure JSON only"""
        system_prompt_po = """You are an expert finance validation assistant.\n
            You will compare an invoice and its corresponding ERP data for quantity, unit price, currency consistency, Flag discrepancies.\n
            You will compare vendor data and its corresponding ERP data for vendor_id, vendor_name, country, currency.
            Evaluate price_difference_percent by comparing prices in invoice against ERP data.
            Evaluate quantity_difference_percent by comparing quantities in invoice against ERP data.
            Evaluatae tax_difference_percent by comparing taxes in invoice against ERP data.
            Evalute disscrepancies in totals, prices, quantities in invoice against ERP data. if discrepancies > 5%, return in output json
            return JSON: {data_differences: {price_difference_percent, quantity_difference_percent, tax_difference_percent}, discrepancies: [{field, total/quantity, discrepancy in %}], issues: [{field, severity, message}]. Categorize severity High/Medium/Low based on the impact. 
            Only output the JSON object. Do NOT include any explanatory text"""
        system_prompt_vendor = """You are an expert finance validation assistant.\n
            You will compare vendor data and its corresponding ERP data for vendor_id, vendor_name, country, currency.
            return JSON: {issues: [{field, severity, message}]. Categorize severity High/Medium/Low based on the impact. Return pure JSON only"""
        
        system_prompt =  system_prompt_po if validation_type == 'po_validation' else system_prompt_vendor
       
        user_text = json.dumps(structured,ensure_ascii=False)  
        # ------------------------------------------------------
        # User message using placeholders
        # ------------------------------------------------------
        user_prompt = (
        "Here is the invoice JSON:\n"
        "{invoice_json}\n\n"
        "Here is the ERP match JSON:\n"
        "{erp_json}\n\n"
        "Analyze the data and produce the JSON response."
        ).format(invoice_json=user_text, erp_json=erp_json)

        user_text = json.dumps(structured,ensure_ascii=False)  
        messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
        AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | LLM call initiated")
        try:
            response = litellm.completion(
            model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
            #model="ollama/llama3.1:latest",
            messages=messages,
            api_base="http://localhost:11434",
            temperature=0.0  # Set low temperature for literal, non-creative translation
            )
            res = response['choices'][0]['message']['content'].strip()
            AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | LLM call completed {res}")
            return res
        except Exception as e:
            error_msg = f"Validation LLM Error via LiteLLM: {e}"
            AUDITOR_LOGGER.error(f"Validation Error: {error_msg}")
            return None
        
def validate_business_logic(state: InvoiceProcessingState) -> Dict:
    """Performs validation checks using Python for math/files and LLM for complex logic."""
    AUDITOR_LOGGER.info("Node: business validation_agent | Starting business validation Agent:")

    struct = state["translated_data"]
    business_issues = []

    po_number = struct.get("po_number") or struct.get("purchase_order") or struct.get("po")
    vendor_id = struct.get("vendor_id")
    data_differences_against_erp = {}
    discrepancies_details = []
    # Fetch PO if present
    po_record = None
    if po_number:
        po_record = fetch_po(str(po_number))
        AUDITOR_LOGGER.info(f"Node: business validation_agent | PO Record fetched:{po_record}")
        if po_record is None:
            business_issues.append({
                    "field": "po_number",
                    "severity": "HIGH",
                    "message": f"PO {po_number} not found in ERP"
            })
        else:
            try:
                eval_res = business_validation_llm_call(struct, po_record, 'po_validation')
                AUDITOR_LOGGER.info(f"Node: business validation_agent | eval_res invoice:{eval_res}")
                data = json.loads(eval_res)
                AUDITOR_LOGGER.info(f"Node: business validation_agent | data invoice:{data}")
                data_differences_against_erp = data["data_differences"]
                for i in data.get('issues', []):
                    business_issues.append(i)
                AUDITOR_LOGGER.info(f"Node: business validation_agent | business_issues:{business_issues}")
                for i in data.get('discrepancies', []):
                    discrepancies_details.append(i)
            
            except Exception as e:
                error_msg = f"Validation LLM Error via LiteLLM: {e}"
                AUDITOR_LOGGER.error(error_msg)

        # Fetch vendor if present
    vendor_record = None
    AUDITOR_LOGGER.info(f"Node: business validation_agent | vendor_id in invoice:{vendor_id}")
    if vendor_id is not None:   
            AUDITOR_LOGGER.info(f"Node: business validation_agent | inside vendor_id in invoice:{vendor_id}")
            vendor_record = fetch_vendor(str(vendor_id))
            if vendor_record is None:
                business_issues.append({
                        "field": "vendor_id",
                        "severity": "HIGH",
                        "message": f"Vendor {vendor_id} not found in ERP"
                    })
            else:
            
                try:
                    eval_res = business_validation_llm_call(struct, vendor_record, 'vendor_validation')
                    AUDITOR_LOGGER.info(f"eval_res Error: {eval_res}")
                    data = json.loads(eval_res)
                    for i in data.get('issues', []):
                        business_issues.append(i)
                except Exception as e:
                    error_msg = f"Validation LLM Error via LiteLLM: {e}"
                    AUDITOR_LOGGER.ERROR(error_msg)
    AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | execution completed: business_issues: {business_issues}\n data_differences_against_erp:{data_differences_against_erp}\n discrepancies_details: {discrepancies_details}")
    return {"data_differences_against_erp": data_differences_against_erp, "business_validation_issues": business_issues, "discrepancies_details": discrepancies_details}
   


'''def validate_business_logic(state: InvoiceProcessingState) -> Dict:
    # Logic: Call Mock ERP API (FastAPI) to verify PO/price/qty.
    print("--- 5. Validating Business Logic (ERP Check)...")
    # Simulation: Assume ERP check fails due to price discrepancy
    new_results = state['validation_results']
    new_results.update({"erp_match": False, "price_discrepancy": True})
    return {"validation_results": new_results, "audit_score": 0.75}'''
'''
def reporting(state: InvoiceProcessingState) -> Dict:
    # Logic: Generate JSON/HTML reports, determine final recommendation.
    print("--- 6. Generating Report...")
    
    # Determine recommendation based on audit score/discrepancies
    if not state['validation_results']['erp_match'] or not state['validation_results']['completeness_ok']:
        recommendation = "Manual Review"
    else:
        recommendation = "Approve"
        
    return {"report_path": "/data/reports/INV-123.json", "recommendation": recommendation}'''

# Define the reports folder path
REPORTS_FOLDER = "data/reports"
os.makedirs(REPORTS_FOLDER, exist_ok=True) # Ensure the directory exists

# The Reporting Agent function
def reporting(state: InvoiceProcessingState) -> Dict:
    # 1. Determine final recommendation (logic based on validation_results)
    invoice_data = state['translated_data']
    translation_confidence = state['translation_confidence']
    rules = state['validation_rules_yaml']
    data_differences_against_erp = state['data_differences_against_erp']
    discrepancies = state["discrepancies_details"]
    system_prompt = """You are an expert report generation assistant. Your task is to check the provid recommnedation and .
        =**RULES & INSTRUCTIONS:**\n
            You will compare data_differences_against_erp json against tolerances field in rules
            **Use below guidelines to generate Recommendation**
                - if totals mismatch beyond tolerance, mark recommendation as manual_review 
                - if the currency in invoice data is not listed in accepted_currencies in rules, mark recommendation as rejected
                - if no discrepancies and translation confidence ≥ 0.95, mark recommendation as Approved
            return {Recommendation: Return pure JSON only\n
            Only output the JSON object. Do NOT include any explanatory text, markdown formatting."""
   

        # ------------------------------------------------------
        # User message using placeholders
        # ------------------------------------------------------
    user_prompt = (
        "Here is the invoice data\n"
        "{invoice_data}\n\n"
        "Here is the validaion rules yaml:\n"
        "{rules}\n\n"
        "Here is the translation_confidence:\n"
        "{translation_confidence}\n\n"
        "Here is the data_differences_against_erp:\n"
        "{data_differences_against_erp}\n\n"
        "Here is the discrepancies list:\n"
        "{discrepancies}\n\n"
        "Analyze the data and produce the JSON response."
        ).format(invoice_data=invoice_data,rules=rules, translation_confidence=translation_confidence, data_differences_against_erp=data_differences_against_erp, discrepancies=discrepancies)

        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
    AUDITOR_LOGGER.info(f"Node: report_agent | LLM call initiated")
    try:
        response = litellm.completion(
        model="ollama/mistral:7b-instruct",  # High-quality model for better instruction following
            #model="ollama/llama3.1:latest",
        messages=messages,
        api_base="http://localhost:11434",
        temperature=0.0  # Set low temperature for literal, non-creative translation
        )
        res = response['choices'][0]['message']['content'].strip()
        AUDITOR_LOGGER.info(f"Node: reporting_agent | LLM call completed {res}")
        data = json.loads(res)
        recommendation = data["Recommendation"]
        file_path_obj = Path(state["file_path"])
        meta_path_pathlib = file_path_obj.with_suffix(".meta.json")

        # 2. Using the 'os.path' module
        # os.path.splitext splits the path into (root, ext)
        base_name, _ = os.path.splitext(state["file_path"])
        meta_path_os = base_name + ".meta.json"
        AUDITOR_LOGGER.info(f"Node: busiess_validation_agent | metadata file name {meta_path_os}")
        meta_data = None
        try:
            with open(meta_path_os, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)

        except FileNotFoundError as e:
            AUDITOR_LOGGER.info(f"Metadata file Error: {e}")
        except json.JSONDecodeError as e:
            AUDITOR_LOGGER.info(f"Error: Failed to parse JSON in {meta_path_os}: {e}")

        # 2. Compile the Final Report Dictionary
        final_report = {
            "invoice_file": state['file_name'],
            "recommendation": recommendation,
            "translated_data": state['translated_data'],
            "validation_issues": state['validation_issues'],
            "business_validation_issues": state['business_validation_issues'],
            "discrepancies_details": state['discrepancies_details'],
            "translation_confidence": state['translation_confidence'],
            "meta_data": meta_data
        }

        # 3. Save the JSON Report
        report_filename = f"{os.path.splitext(state['file_name'])[0]}.json"
        report_path = os.path.join(REPORTS_FOLDER, report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Report saved successfully to: {report_path}")
            
        except IOError as e:
            logging.error(f"Failed to save report: {e}")
            state['errors'].append(f"Reporting Error: Failed to save file.")
            
        # 4. Update the state
        return {"report_path": report_path, "recommendation": recommendation, "status": "REPORTED", "invoice_meta_data": meta_data}
    except Exception as e:
        error_msg = f"Report generationerror via LiteLLM: {e}"
        AUDITOR_LOGGER.error(error_msg)
        return None
    
def decide_to_review(state: InvoiceProcessingState) -> str:
    """Conditional edge logic after initial validation."""
    AUDITOR_LOGGER.info(f"Report generation errordecide_to_review: {state['validation_results']}")
    if state['validation_results'].get('completeness_ok'):
        # If basic data or calculation fails, skip ERP check and go to reporting
        return "validate_business_logic"
    # Otherwise, proceed to external business validation
    else:
        return "reporting"

def decide_to_translate(state: InvoiceProcessingState) -> str:
    """Conditional edge logic after text extractiopn."""
    lang = state.get("detected_language")
    if lang == 'EN':
        # If English, skip the translation agent and jump to validation
        AUDITOR_LOGGER.info("Condition: Language is EN. Skipping translation.")
        return "validate_invoice_data" 
    
    # For any other language (FR, ES, UNKNOWN, etc.), proceed with translation
    AUDITOR_LOGGER.info(f"Condition: Language is {lang}. Proceeding to translation.")
    return "translate_data"

def decide_final_action(state: InvoiceProcessingState) -> str:
    """Conditional edge logic after all validation and reporting."""
    if state['recommendation'] in ["Manual Review", "Reject"]:
        # If human intervention is needed, the process waits in the Streamlit dashboard
        return "human_in_the_loop_wait" 
    else:
        # If approved, proceed to final indexing
        return "rag_indexing"