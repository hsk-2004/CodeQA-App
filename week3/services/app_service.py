import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://localhost:8001/retrieve")
RETRIEVAL_SERVICE_BASE = RETRIEVAL_SERVICE_URL.rsplit("/", 1)[0]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("LLM_MODEL", "codellama:7b")

EVAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "eval_data")
EVAL_RESULTS_FILE = os.path.join(EVAL_DATA_DIR, "results.json")
EVAL_SCORES_FILE = os.path.join(EVAL_DATA_DIR, "manual_scores.json")

CATEGORY_LABELS = {
    "code_explanation": "Explanation",
    "code_retrieval": "Code Retrieval",
    "dependency_understanding": "Dependency Understanding",
    "bug_analysis": "Bug Analysis",
    "code_generation": "Code Generation",
    "refactoring": "Refactoring",
    "rag_based": "RAG-based Question",
    "repo_understanding": "Repository Understanding",
}

app = FastAPI(title="App / Orchestration Service")

# In-memory log of every question actually asked through this running UI
# (separate from the offline Week 4 evaluation dataset in eval_data/).
live_history: list[dict] = []

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

    result = {
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

    live_history.insert(0, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    })

    return result

@app.get("/knowledge-base")
def knowledge_base():
    """Proxies the Retrieval Service's knowledge base contents, so the browser
    (which only talks to this service on port 8000) can display it."""
    response = requests.get(f"{RETRIEVAL_SERVICE_BASE}/knowledge-base")
    response.raise_for_status()
    return response.json()

@app.get("/history")
def history():
    """Live log of every question asked through this UI's Compare tab this session
    (in-memory only -- resets when the container restarts)."""
    return {"total": len(live_history), "entries": live_history}

@app.get("/metrics")
def metrics():
    """Week 4 offline multi-model evaluation results, broken down per category
    (Explanation, Code Retrieval, Dependency Understanding, Bug Analysis,
    Code Generation, Refactoring, RAG-based Question)."""
    if not os.path.exists(EVAL_RESULTS_FILE):
        return {"available": False, "message": "No evaluation results found. Run week4/run_evaluation.py first."}

    with open(EVAL_RESULTS_FILE) as f:
        results = json.load(f)

    models = sorted({m for entry in results for m in entry["models"]})

    overall = defaultdict(lambda: {"latency": [], "tokens": []})
    by_category = defaultdict(lambda: defaultdict(lambda: {"latency": [], "tokens": []}))
    category_order = []

    for entry in results:
        cat = entry["category"]
        if cat not in category_order:
            category_order.append(cat)
        for model, res in entry["models"].items():
            if res.get("latency_seconds") is not None:
                overall[model]["latency"].append(res["latency_seconds"])
                by_category[cat][model]["latency"].append(res["latency_seconds"])
            if res.get("eval_count") is not None:
                overall[model]["tokens"].append(res["eval_count"])
                by_category[cat][model]["tokens"].append(res["eval_count"])

    def summarize(d):
        lat, tok = d["latency"], d["tokens"]
        return {
            "avg_latency_seconds": round(sum(lat) / len(lat), 2) if lat else None,
            "avg_tokens": round(sum(tok) / len(tok), 1) if tok else None,
            "questions_answered": len(lat),
        }

    # Merge in manual correctness/hallucination scoring, if available.
    scores_by_qid = {}
    if os.path.exists(EVAL_SCORES_FILE):
        with open(EVAL_SCORES_FILE) as f:
            scores_by_qid = {q["id"]: q["scores"] for q in json.load(f)["questions"]}

    def quality_summary(entries_for_model_in_scope):
        """entries_for_model_in_scope: list of (question_id, model) pairs."""
        total = len(entries_for_model_in_scope)
        if total == 0:
            return {"correct": 0, "partial": 0, "incorrect_or_no_answer": 0,
                    "accuracy_pct": None, "hallucination_rate_pct": None}
        correct = partial = bad = hallucinated = 0
        for qid, model in entries_for_model_in_scope:
            s = scores_by_qid.get(qid, {}).get(model)
            if not s:
                continue
            if s["verdict"] == "correct":
                correct += 1
            elif s["verdict"] == "partial":
                partial += 1
            else:
                bad += 1
            if s.get("hallucinated"):
                hallucinated += 1
        return {
            "correct": correct,
            "partial": partial,
            "incorrect_or_no_answer": bad,
            "accuracy_pct": round(100 * correct / total, 1),
            "hallucination_rate_pct": round(100 * hallucinated / total, 1),
        }

    overall_quality = {
        model: quality_summary([(e["id"], model) for e in results])
        for model in models
    }
    by_category_quality = {
        cat: {
            model: quality_summary([(e["id"], model) for e in results if e["category"] == cat])
            for model in models
        }
        for cat in category_order
    }

    return {
        "available": True,
        "total_questions": len(results),
        "models": models,
        "overall": {
            model: {**summarize(overall[model]), **overall_quality[model]}
            for model in models
        },
        "by_category": {
            cat: {
                "label": CATEGORY_LABELS.get(cat, cat),
                "models": {
                    model: {**summarize(by_category[cat][model]), **by_category_quality[cat][model]}
                    for model in models
                },
            }
            for cat in category_order
        },
    }

@app.get("/evaluation")
def evaluation():
    """Full per-question, per-model breakdown: answer text, latency, tokens,
    manual correctness verdict, and hallucination flag -- everything needed
    to inspect the Week 4 evaluation in detail."""
    if not os.path.exists(EVAL_RESULTS_FILE):
        return {"available": False, "message": "No evaluation results found. Run week4/run_evaluation.py first."}

    with open(EVAL_RESULTS_FILE) as f:
        results = json.load(f)

    scores_by_qid = {}
    metric_definitions = {}
    if os.path.exists(EVAL_SCORES_FILE):
        with open(EVAL_SCORES_FILE) as f:
            scores_data = json.load(f)
            scores_by_qid = {q["id"]: q["scores"] for q in scores_data["questions"]}
            metric_definitions = scores_data.get("metric_definitions", {})

    models = sorted({m for entry in results for m in entry["models"]})

    questions = []
    for entry in results:
        model_data = {}
        for model in models:
            res = entry["models"].get(model, {})
            score = scores_by_qid.get(entry["id"], {}).get(model, {})
            model_data[model] = {
                "answer": res.get("answer", ""),
                "latency_seconds": res.get("latency_seconds"),
                "tokens": res.get("eval_count"),
                "verdict": score.get("verdict", "unscored"),
                "hallucinated": score.get("hallucinated", False),
                "notes": score.get("notes", ""),
            }
        questions.append({
            "id": entry["id"],
            "category": entry["category"],
            "category_label": CATEGORY_LABELS.get(entry["category"], entry["category"]),
            "question": entry["question"],
            "retrieved_from": entry.get("retrieved_from", []),
            "models": model_data,
        })

    return {
        "available": True,
        "models": models,
        "metric_definitions": metric_definitions,
        "questions": questions,
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL}

# Serve the frontend UI (static/index.html) at the root path.
# Mounted last so it doesn't shadow the API routes above.
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
