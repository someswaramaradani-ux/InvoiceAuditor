# agents/rag_agents/generator.py
from agents.litellm_client import LiteLLMClient
from agents.rag_agents.retriever import retrieve
from agents.rag_agents.reranker import rerank, load_chunks
from agents.rag_agents.utils import load_json
from typing import List, Dict
import json

client = LiteLLMClient()

PROMPT_SYSTEM = (
    "You are InvoiceAuditRAG — an assistant that answers questions about invoices and audit reports. "
    "You MUST base your answer only on the provided source excerpts. "
    "First, list which report(s) and chunk(s) you used (source_report and chunk_id). "
    "If you cannot find an answer in the sources, say 'INSUFFICIENT_DATA' and explain what is missing. "
    "Be concise and provide a short recommended action (Approve / Manual Review / Reject) where appropriate."
)

PROMPT_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Context excerpts:\n{excerpts}\n\n"
    "Provide a concise answer that cites the excerpt chunk_ids. Do not hallucinate.\n"
    "Response should be JSON with fields: answer (string), sources (list of {{source_report,chunk_id}}), confidence (0-1), recommendation.\n"
)

def format_excerpts(candidates: List[Dict], max_chars_per_excerpt=1000):
    out = []
    for c in candidates:
        txt = c["text"][:max_chars_per_excerpt].replace("\n"," ")
        out.append(f"CHUNK_ID: {c['meta']['chunk_id']} | REPORT: {c['meta']['source_report']} | SCORE: {c.get('rerank_score', c.get('score'))}\nTEXT: {txt}")
    return "\n\n".join(out)

def answer_question(question: str, top_k: int = 5) -> Dict:
    # 1) retrieve
    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        return {"error": "no index or no documents found"}
    # 2) rerank
    reranked = rerank(question, retrieved, top_k=top_k)
    if not reranked:
        # fallback to retrieved with minimal formatting
        reranked = [{"meta": r["meta"], "text": ""} for r in retrieved]
    # 3) format context
    excerpts = format_excerpts(reranked, max_chars_per_excerpt=1200)
    user_prompt = PROMPT_USER_TEMPLATE.format(question=question, excerpts=excerpts)
    # 4) call LLM
    resp = client.generate(PROMPT_SYSTEM, user_prompt, max_tokens=512, temperature=0.0)
    # attempt to parse JSON out of the response
    text = resp.get("text","")
    try:
        parsed = json.loads(text)
        return {"answer": parsed, "raw": text, "used_chunks": [ {"chunk_id": r["meta"]["chunk_id"], "source_report": r["meta"]["source_report"]} for r in reranked ]}
    except Exception:
        # return the text and the chunks used
        return {"answer": {"answer_text": text}, "raw": text, "used_chunks": [ {"chunk_id": r["meta"]["chunk_id"], "source_report": r["meta"]["source_report"]} for r in reranked ]}
