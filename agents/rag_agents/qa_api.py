# agents/rag_agents/qa_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.rag_agents.generator import answer_question
import uvicorn

app = FastAPI(title="Invoice Auditor RAG QA")

class QARequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/qa")
def qa(req: QARequest):
    if not req.question or len(req.question.strip())==0:
        raise HTTPException(status_code=400, detail="Empty question")
    out = answer_question(req.question, top_k=req.top_k)
    return out

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
