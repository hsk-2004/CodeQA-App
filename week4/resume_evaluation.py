"""
Resumes run_evaluation.py from wherever it left off, using the existing
results.json (which already has prompts/retrieved_from precomputed, and
whatever model answers were already saved). Only asks models that are
missing for each question -- skips anything already done.

Includes a per-request timeout so a single stuck call can't hang forever;
on timeout it records an error placeholder and moves on to the next question.
"""
import json
import time
import requests

RESULTS_FILE = "results.json"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODELS = ["codellama:7b", "starcoder2:3b", "phi3:mini"]
TIMEOUT_SECONDS = 180  # give up on a single answer after 3 minutes


def ask_model(model: str, prompt: str) -> dict:
    start = time.time()
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=TIMEOUT_SECONDS
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
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        return {
            "answer": "[TIMED OUT - no response within " + str(TIMEOUT_SECONDS) + "s]",
            "latency_seconds": round(elapsed, 2),
            "prompt_eval_count": None,
            "eval_count": None,
        }


def main():
    with open(RESULTS_FILE) as f:
        prepared = json.load(f)

    for model in MODELS:
        print(f"\n########## MODEL: {model} ##########")
        for entry in prepared:
            if model in entry["models"]:
                continue  # already done, skip
            print(f"  Q{entry['id']} [{entry['category']}]: {entry['question']}")
            result = ask_model(model, entry["prompt"])
            entry["models"][model] = result
            print(f"    -> done in {result['latency_seconds']}s, {result.get('eval_count')} tokens")

            with open(RESULTS_FILE, "w") as f:
                json.dump(prepared, f, indent=2)

    print("\nAll done (or timed-out entries recorded). Results saved to results.json")


if __name__ == "__main__":
    main()
