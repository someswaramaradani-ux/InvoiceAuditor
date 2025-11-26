"""
Provider-agnostic LiteLLM client wrapper (OpenAI-free by default).

Supported providers:
 - mock   : deterministic local mock (default)
 - local  : REST API (LITELLM_URL) - your local LiteLLM runtime that accepts JSON {prompt: "..."}
 - bedrock: AWS Bedrock via boto3 (optional; boto3 required)

Environment variables:
 - LITELLM_PROVIDER (mock | local | bedrock)   default: mock
 - LITELLM_URL      (for local)                e.g. http://localhost:8080/generate
 - AWS_REGION, BEDROCK_MODEL_ID (for bedrock)

Usage:
  from agents.litellm_client import LiteLLMClient
  client = LiteLLMClient()
  client.translate(text, source_lang='es', target_lang='en')
"""

import os
import json
import time
from typing import Any, Dict, Optional

DEFAULT_PROVIDER = os.getenv("LITELLM_PROVIDER", "mock")

# lazy imports for optional providers
_boto3 = None
_requests = None
try:
    import requests as _requests
except Exception:
    _requests = None

try:
    import boto3 as _boto3
except Exception:
    _boto3 = None


class LiteLLMClient:
    def __init__(self, provider: Optional[str] = None, url: Optional[str] = None):
        """
        provider: 'mock' | 'local' | 'bedrock'
        url: for local REST provider (overrides env LITELLM_URL)
        """
        self.provider = (provider or os.getenv("LITELLM_PROVIDER") or DEFAULT_PROVIDER).lower()
        self.url = url or os.getenv("LITELLM_URL")
        self._init_provider_clients()

    def _init_provider_clients(self):
        if self.provider == "bedrock":
            if _boto3 is None:
                raise RuntimeError("boto3 is required for bedrock provider. Install boto3.")
            # we will create client lazily
            self._bedrock_client = None
        elif self.provider == "local":
            if _requests is None:
                raise RuntimeError("requests package is required for local provider. Install requests.")
        # mock requires nothing

    # ---------------------------
    # Low-level provider call
    # ---------------------------
    def _call_provider(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        t0 = time.time()
        if self.provider == "mock":
            return self._mock_call(prompt, max_tokens, temperature)
        if self.provider == "local":
            return self._local_call(prompt, max_tokens, temperature)
        if self.provider == "bedrock":
            return self._bedrock_call(prompt, max_tokens, temperature)
        # fallback to mock
        return self._mock_call(prompt, max_tokens, temperature)

    def _mock_call(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Deterministic mocked responses suitable for offline demos.
        This is intentionally simple; adjust the heuristics as needed.
        """
        # short heuristics to detect the requested task
        t0 = time.time()
        low = prompt.lower()
        if "translate" in low and "translate to" in low:
            # echo the "text" part after the last newline
            snippet = prompt.splitlines()[-1][:4000]
            return {"text": json.dumps({"translated_text": snippet, "source_lang": "auto", "confidence": 0.95})}
        if "extract common invoice fields" in low or "invoice extraction" in low:
            # produce small structured JSON by heuristics
            inv = "INV-0001"
            total = "100.00"
            resp = {
                "invoice_number": inv,
                "date": None,
                "vendor": "ACME Co.",
                "vendor_id": None,
                "currency": "USD",
                "total": total,
                "lines": [
                    {"item_code": "SKU-1", "description": "Widget", "qty": 2, "unit_price": 25.0, "total": 50.0},
                    {"item_code": "SKU-2", "description": "Service", "qty": 1, "unit_price": 50.0, "total": 50.0},
                ],
            }
            return {"text": json.dumps(resp)}
        if "evaluate this structured invoice" in low or "forensic invoice auditor" in low:
            # simple evaluation logic that returns issues and suggestion
            try:
                payload_start = prompt.find("{")
                structured = json.loads(prompt[payload_start:]) if payload_start >= 0 else {}
            except Exception:
                structured = {}
            issues = []
            suggestion = "Approve"
            if not structured.get("invoice_number"):
                issues.append({"code": "missing_invoice_number", "field": "invoice_number", "severity": "MED", "message": "Missing invoice_number"})
            try:
                total = float(structured.get("total", 0) or 0)
                lines_sum = sum(float(l.get("total", 0) or 0) for l in structured.get("lines", []))
                if abs(lines_sum - total) > 0.01:
                    issues.append({"code": "total_mismatch", "field": "total", "severity": "MED", "message": f"Total {total} != sum lines {lines_sum}"})
                    suggestion = "Manual Review"
            except Exception:
                issues.append({"code": "invalid_total", "field": "total", "severity": "HIGH", "message": "Invalid total"})
                suggestion = "Manual Review"
            return {"text": json.dumps({"issues": issues, "suggestion": suggestion})}

        # default
        return {"text": f"MOCK ECHO: {prompt[:200]}", "raw": {"elapsed": time.time() - t0}}

    def _local_call(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        POST JSON to a local LiteLLM server. The endpoint contract is up to you.
        Default expects JSON body: {"prompt": "...", "max_tokens": ..., "temperature": ...}
        and a JSON response with {"text": "..."}.
        """
        if not self.url:
            raise RuntimeError("LITELLM_URL is not set for local provider.")
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        resp = _requests.post(self.url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # expected to contain 'text'
        if isinstance(data, dict) and "text" in data:
            return {"text": data["text"], "raw": data}
        # fallback
        return {"text": str(data), "raw": data}

    def _bedrock_call(self, prompt: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        """
        Minimal Bedrock invocation. The exact body depends on the model.
        This method expects environment variables:
          AWS_REGION, BEDROCK_MODEL_ID
        """
        if _boto3 is None:
            raise RuntimeError("boto3 is required for bedrock provider.")
        region = os.getenv("AWS_REGION", "us-east-1")
        model_id = os.getenv("BEDROCK_MODEL_ID")
        if not model_id:
            raise RuntimeError("BEDROCK_MODEL_ID env var not set.")
        if self._bedrock_client is None:
            self._bedrock_client = _boto3.client("bedrock", region_name=region)
        # NOTE: adapt the body to your model's expected input format
        body = {"input": prompt, "max_tokens": max_tokens, "temperature": temperature}
        response = self._bedrock_client.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body))
        # response['body'] is a streaming HTTPResponse-like object for some SDKs
        body_bytes = response.get("body")
        if hasattr(body_bytes, "read"):
            body_text = body_bytes.read().decode("utf-8")
        else:
            body_text = body_bytes.decode("utf-8") if isinstance(body_bytes, (bytes, bytearray)) else str(body_bytes)
        # model may return JSON or plain text
        try:
            parsed = json.loads(body_text)
            # try to extract a sensible return text
            if isinstance(parsed, dict) and "output" in parsed:
                return {"text": parsed["output"], "raw": parsed}
            return {"text": body_text, "raw": parsed}
        except Exception:
            return {"text": body_text, "raw": body_text}

    # ---------------------------
    # High-level helpers (used by nodes)
    # ---------------------------
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generic generate interface: merges system and user prompts into a single prompt.
        Returns dict with 'text' (string) and optional 'raw'.
        """
        prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n"
        return self._call_provider(prompt, max_tokens=max_tokens, temperature=temperature)

    def translate(self, text: str, source_lang: Optional[str] = None, target_lang: str = "en") -> Dict[str, Any]:
        """
        Translate text -> returns dict that should contain: translated_text, source_lang, confidence
        """
        system = f"You are a professional translator. Translate the following text to {target_lang} preserving numbers, currencies and tables. Return JSON: {{translated_text, source_lang, confidence}}"
        user = text[:8000]
        resp = self.generate(system, user, max_tokens=1024, temperature=0.0)
        txt = resp.get("text", "")
        # try to parse JSON
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict) and "translated_text" in parsed:
                return parsed
        except Exception:
            # fallback to returning raw text
            return {"translated_text": txt, "source_lang": source_lang or "unknown", "confidence": 0.6}
        return {"translated_text": txt, "source_lang": source_lang or "unknown", "confidence": 0.6}

    def extract_structured(self, text: str) -> Dict[str, Any]:
        """
        Ask the model to return structured invoice JSON.
        Expected output: JSON with fields invoice_number, date, vendor, currency, total, lines[]
        """
        system = "You are an invoice extraction assistant. Extract common invoice fields as JSON: invoice_number, date, vendor, vendor_id, currency, total, lines (list of {item_code, description, qty, unit_price, total}). Return pure JSON only."
        user = text[:12000]
        resp = self.generate(system, user, max_tokens=1024, temperature=0.0)
        txt = resp.get("text", "")
        try:
            parsed = json.loads(txt)
            return parsed
        except Exception:
            # fallback: return a minimal structure
            return {"invoice_number": None, "date": None, "vendor": None, "currency": None, "total": None, "lines": []}

    def evaluate_invoice(self, structured: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM to evaluate a structured invoice and return issues + suggestion.
        Expected output: JSON {issues: [...], suggestion: "..."}
        """
        system = "You are a forensic invoice auditor. Evaluate this structured invoice JSON for likely errors and return JSON: {issues: [{code, field, severity, message}], suggestion}"
        user = json.dumps(structured)[:12000]
        resp = self.generate(system, user, max_tokens=1024, temperature=0.0)
        txt = resp.get("text", "")
        try:
            parsed = json.loads(txt)
            return parsed
        except Exception:
            return {"issues": [], "suggestion": None, "raw": txt}
