import json
import math
import requests

KB_FILE = "knowledge_base.json"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "codellama:7b"
TOP_K = 2

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

def retrieve_relevant_chunks(question: str, knowledge_base: list[dict], top_k: int = TOP_K) -> list[dict]:
    question_embedding = get_embedding(question)
    scored = []
    for item in knowledge_base:
        score = cosine_similarity(question_embedding, item["embedding"])
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]

def ask_with_context(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"File: {c['file']}\n{c['text']}" for c in context_chunks
    )
    prompt = f"""Use the following code context to answer the question.

Context:
{context_text}

Question: {question}
Answer:"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

def ask_without_context(question: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": LLM_MODEL, "prompt": question, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    with open(KB_FILE) as f:
        knowledge_base = json.load(f)

    question = "Which file handles user authentication and how does it work?"

    print("=== WITHOUT RAG (no context) ===")
    print(ask_without_context(question))

    print("\n=== WITH RAG (retrieved context) ===")
    relevant_chunks = retrieve_relevant_chunks(question, knowledge_base)
    print("Retrieved chunks from:", [c["file"] for c in relevant_chunks])
    print(ask_with_context(question, relevant_chunks))
