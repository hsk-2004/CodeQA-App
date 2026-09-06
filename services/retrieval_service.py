import json
import math
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

KB_FILE = os.path.join(os.path.dirname(__file__), "..", "knowledge_base.json")
EMBED_MODEL = "nomic-embed-text"
TOP_K = 2

app = FastAPI(title="Retrieval Service")

with open(KB_FILE) as f:
    knowledge_base = json.load(f)

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

def get_embedding(text: str) -> list[float]:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

@app.post("/retrieve")
def retrieve(req: QueryRequest):
    question_embedding = get_embedding(req.question)
    scored = []
    for item in knowledge_base:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [
        {"file": item["file"], "text": item["text"], "score": score}
        for score, item in scored[:req.top_k]
    ]
    return {"chunks": top_chunks}

@app.get("/health")
def health():
    return {"status": "ok", "chunks_loaded": len(knowledge_base)}
