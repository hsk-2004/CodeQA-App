import os
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001/retrieve")
RETRIEVAL_SERVICE_BASE = RETRIEVAL_SERVICE_URL.rsplit("/", 1)[0]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("LLM_MODEL", "codellama:7b")

app = FastAPI(title="App / Orchestration Service")

class AskRequest(BaseModel):
    question: str

def call_retrieval_service(question: str) -> list[dict]:
    response = requests.post(RETRIEVAL_SERVICE_URL, json={"question": question})
    response.raise_for_status()
    return response.json()["chunks"]

def build_prompt_with_context(question: str, chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"File: {c['file']}\n{c['text']}" for c in chunks
    )
    return f"""Use the following code context to answer the question.

Context:
{context_text}

Question: {question}
Answer:"""

def call_llm(prompt: str) -> dict:
    start = time.time()
    response = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
    )
    elapsed = round(time.time() - start, 2)
    response.raise_for_status()
    data = response.json()
    return {"answer": data.get("response", ""), "latency_seconds": elapsed}

@app.post("/ask")
def ask(req: AskRequest):
    chunks = call_retrieval_service(req.question)
    prompt = build_prompt_with_context(req.question, chunks)
    result = call_llm(prompt)
    return {
        "question": req.question,
        "retrieved_from": [c["file"] for c in chunks],
        "answer": result["answer"],
        "latency_seconds": result["latency_seconds"],
    }

@app.post("/compare")
def compare(req: AskRequest):
    """Answers the question both WITHOUT retrieval (raw model) and WITH retrieval (RAG),
    so the difference can be seen side by side."""
    without_rag = call_llm(req.question)

    chunks = call_retrieval_service(req.question)
    prompt_with_context = build_prompt_with_context(req.question, chunks)
    with_rag = call_llm(prompt_with_context)

    return {
        "question": req.question,
        "without_rag": {
            "answer": without_rag["answer"],
            "latency_seconds": without_rag["latency_seconds"],
        },
        "with_rag": {
            "answer": with_rag["answer"],
            "latency_seconds": with_rag["latency_seconds"],
            "retrieved_from": [c["file"] for c in chunks],
            "retrieved_chunks": [{"file": c["file"], "text": c["text"]} for c in chunks],
        },
        "model": LLM_MODEL,
    }

@app.get("/knowledge-base")
def knowledge_base():
    """Proxies the Retrieval Service's knowledge base contents, so the browser
    (which only talks to this service on port 8000) can display it."""
    response = requests.get(f"{RETRIEVAL_SERVICE_BASE}/knowledge-base")
    response.raise_for_status()
    return response.json()

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}

# Serve the frontend UI (static/index.html) at the root path.
# Mounted last so it doesn't shadow the API routes above.
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
