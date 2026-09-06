from fastapi import FastAPI
from pydantic import BaseModel
import requests

RETRIEVAL_SERVICE_URL = "http://localhost:8001/retrieve"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "codellama:7b"

app = FastAPI(title="App / Orchestration Service")

class AskRequest(BaseModel):
    question: str

def call_retrieval_service(question: str) -> list[dict]:
    response = requests.post(RETRIEVAL_SERVICE_URL, json={"question": question})
    response.raise_for_status()
    return response.json()["chunks"]

def call_llm_service(question: str, chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"File: {c['file']}\n{c['text']}" for c in chunks
    )
    prompt = f"""Use the following code context to answer the question.

Context:
{context_text}

Question: {question}
Answer:"""
    response = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

@app.post("/ask")
def ask(req: AskRequest):
    chunks = call_retrieval_service(req.question)
    answer = call_llm_service(req.question, chunks)
    return {
        "question": req.question,
        "retrieved_from": [c["file"] for c in chunks],
        "answer": answer
    }

@app.get("/health")
def health():
    return {"status": "ok"}
