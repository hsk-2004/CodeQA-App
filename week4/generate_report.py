"""
Reads results.json (produced by run_evaluation.py) and generates REPORT.md:
a per-question breakdown of every model's answer, latency, and token usage,
plus aggregate stats per model.

Note: correctness/hallucination scoring is NOT automated here -- the assignment
requires human judgment for those (Week 4 Exercise 3). This script fills in
everything objectively measurable (latency, tokens) and lays out the answers
side-by-side so a human reviewer can score correctness/relevance/hallucination
directly in the table.
"""
import json
from collections import defaultdict

RESULTS_FILE = "results.json"
REPORT_FILE = "REPORT.md"


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    models = sorted({m for entry in results for m in entry["models"]})

    lines = []
    lines.append("# Week 4 Evaluation Report\n")
    lines.append(f"Questions evaluated: **{len(results)}**  |  Models compared: **{', '.join(models)}**\n")

    # ---- Aggregate stats ----
    lines.append("## Aggregate Performance (objective metrics)\n")
    lines.append("| Model | Avg Latency (s) | Avg Tokens Generated | Total Latency (s) |")
    lines.append("|---|---|---|---|")

    agg = defaultdict(lambda: {"latency": [], "tokens": []})
    for entry in results:
        for model, res in entry["models"].items():
            if res.get("latency_seconds") is not None:
                agg[model]["latency"].append(res["latency_seconds"])
            if res.get("eval_count") is not None:
                agg[model]["tokens"].append(res["eval_count"])

    for model in models:
        lat = agg[model]["latency"]
        tok = agg[model]["tokens"]
        avg_lat = round(sum(lat) / len(lat), 2) if lat else "N/A"
        avg_tok = round(sum(tok) / len(tok), 1) if tok else "N/A"
        total_lat = round(sum(lat), 2) if lat else "N/A"
        lines.append(f"| {model} | {avg_lat} | {avg_tok} | {total_lat} |")

    lines.append("")

    # ---- Per-question breakdown ----
    lines.append("## Per-Question Results\n")
    for entry in results:
        lines.append(f"### Q{entry['id']} [{entry['category']}]: {entry['question']}\n")
        lines.append(f"*Retrieved context from: {', '.join(entry['retrieved_from'])}*\n")
        lines.append("| Model | Latency (s) | Tokens | Answer |")
        lines.append("|---|---|---|---|")
        for model in models:
            res = entry["models"].get(model)
            if not res:
                lines.append(f"| {model} | - | - | *(not run)* |")
                continue
            answer = res["answer"].replace("\n", " ").replace("|", "\\|").strip()
            if len(answer) > 500:
                answer = answer[:500] + "..."
            lines.append(f"| {model} | {res['latency_seconds']} | {res['eval_count']} | {answer} |")
        lines.append("")

    lines.append("## Conclusion\n")
    lines.append("*(To be filled in based on manual review of correctness/hallucination "
                  "for each answer above, combined with the objective latency/token stats.)*\n")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written to {REPORT_FILE}")


if __name__ == "__main__":
    main()
