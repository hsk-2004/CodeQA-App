import json
import math
import time
import requests

KB_FILE = "../week3/knowledge_base.json"
QUESTIONS_FILE = "questions.json"
RESULTS_FILE = "results.json"

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
TOP_K = 2

MODELS = ["codellama:7b", "starcoder2:3b", "phi3:mini"]


def get_embedding(text: str) -> list[float]:
    response = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def retrieve_relevant_chunks(question: str, knowledge_base: list[dict], top_k: int = TOP_K) -> list[dict]:
    question_embedding = get_embedding(question)
    scored = [(cosine_similarity(question_embedding, item["embedding"]), item) for item in knowledge_base]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored[:top_k]]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_text = "\n\n".join(f"File: {c['file']}\n{c['text']}" for c in chunks)
    return f"""Use the following code context to answer the question.

Context:
{context_text}

Question: {question}
Answer:"""


def ask_model(model: str, prompt: str) -> dict:
    start = time.time()
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": False}
    )
    elapsed = time.time() - start
    response.raise_for_status()
    data = response.json()
    return {
        "answer": data.get("response", ""),
        "latency_seconds": round(elapsed, 2),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
    }


def main():
    with open(KB_FILE) as f:
        knowledge_base = json.load(f)
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)

    # Pre-compute retrieval once per question (same context is used for every model).
    prepared = []
    for q in questions:
        chunks = retrieve_relevant_chunks(q["question"], knowledge_base)
        prepared.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "retrieved_from": [c["file"] for c in chunks],
            "prompt": build_prompt(q["question"], chunks),
            "models": {}
        })

    # Loop model-first so each model loads into memory once and stays warm,
    # instead of reloading a different model for every question.
    for model in MODELS:
        print(f"\n########## MODEL: {model} ##########")
        for entry in prepared:
            print(f"  Q{entry['id']} [{entry['category']}]: {entry['question']}")
            result = ask_model(model, entry["prompt"])
            entry["models"][model] = result
            print(f"    -> done in {result['latency_seconds']}s, {result['eval_count']} tokens")

            with open(RESULTS_FILE, "w") as f:
                json.dump(prepared, f, indent=2)

    print(f"\nAll done. Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
