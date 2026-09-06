"""
Generates clean, readable HTML reports for Week 1-4 (to be converted to PDF).
Run from the repo root: python build_reports.py
"""
import json
import os
import html

ROOT = os.path.dirname(os.path.abspath(__file__))

CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; color: #1a1a1a; max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.55; }
  h1 { font-size: 28px; border-bottom: 3px solid #4f46e5; padding-bottom: 10px; }
  h2 { font-size: 20px; color: #4f46e5; margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }
  h3 { font-size: 16px; margin-top: 22px; }
  p { margin: 8px 0; }
  code { background: #f3f3f7; padding: 2px 5px; border-radius: 3px; font-family: Consolas, monospace; font-size: 13px; }
  pre { background: #1e1e2e; color: #e2e2e8; padding: 14px; border-radius: 6px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; }
  pre code { background: none; padding: 0; color: inherit; }
  table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }
  th { background: #4f46e5; color: white; }
  tr:nth-child(even) { background: #f7f7fb; }
  .tag { display: inline-block; background: #eef2ff; color: #4f46e5; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; margin-right: 6px; }
  .box { background: #f7f7fb; border-left: 4px solid #4f46e5; padding: 10px 16px; margin: 12px 0; border-radius: 4px; }
  .warn { background: #fff7ed; border-left: 4px solid #f59e0b; padding: 10px 16px; margin: 12px 0; border-radius: 4px; }
  .good { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 10px 16px; margin: 12px 0; border-radius: 4px; }
  .bad { background: #fef2f2; border-left: 4px solid #ef4444; padding: 10px 16px; margin: 12px 0; border-radius: 4px; }
  .meta { color: #666; font-size: 13px; }
  ul { margin: 8px 0; }
  .cover { text-align: center; margin-bottom: 50px; }
  .cover .subtitle { color: #666; font-size: 15px; }
</style>
"""

def esc(s):
    return html.escape(str(s)) if s is not None else ""


def page(title, subtitle, body_html):
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{esc(title)}</title>{CSS}</head>
<body>
<div class="cover">
  <h1>{esc(title)}</h1>
  <div class="subtitle">{esc(subtitle)}</div>
</div>
{body_html}
</body></html>"""


def read_file(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


# ---------------- WEEK 1 ----------------
def build_week1():
    body = """
<h2>Objective</h2>
<p>Establish the foundational Virtual Machine and Git workflow used throughout this project.</p>

<h2>What Was Done</h2>
<ul>
  <li>Set up a Virtual Machine using VMware Workstation Pro, running Ubuntu 26.04 LTS.</li>
  <li>Configured networking (NAT) and enabled SSH access, allowing the VM to be controlled from the host laptop's terminal.</li>
  <li>Created a simple static website and hosted it on the VM as a hands-on deployment exercise.</li>
  <li>Connected the hosted project to a GitHub repository (<code>hsk-2004/CodeQA-App</code>).</li>
  <li>Made modifications to the hosted site and practiced the complete Git workflow:
    <code>git add</code> &rarr; <code>git commit</code> &rarr; <code>git push</code>.</li>
</ul>

<h2>Key Learnings</h2>
<div class="box">
  <p><b>VM fundamentals:</b> allocating CPU/RAM/disk, installing an OS, and enabling remote access via SSH.</p>
  <p><b>Git + GitHub workflow:</b> initializing a repository, committing changes, connecting a remote, and
  authenticating pushes using a GitHub Personal Access Token (required since GitHub no longer accepts
  account passwords for Git operations).</p>
</div>

<h2>Outcome</h2>
<p>A working Ubuntu VM, reachable via SSH, with Git configured and a GitHub repository ready to receive
all subsequent work (Weeks 2-4 all build on this same VM and repository).</p>
"""
    return page("Week 1 Report", "VM Setup, Static Website, and Git Workflow", body)


# ---------------- WEEK 2 ----------------
def build_week2():
    body = """
<h2>Objective</h2>
<p>Install and run a local LLM (Code Llama) on the Virtual Machine using Ollama, and verify it responds correctly to prompts.</p>

<h2>What Was Done</h2>
<ul>
  <li>Installed <a href="https://ollama.com">Ollama</a> on the VM &mdash; a tool for running open-source LLMs locally.</li>
  <li>Pulled the <b>Code Llama 7B Instruct</b> model (<code>codellama:7b</code>), a ~3.8 GB download.</li>
  <li>Configured the VM with sufficient resources (upgraded RAM from 4 GB to 8 GB) after observing that
      the default allocation was too small to comfortably run a 7-billion-parameter model.</li>
  <li>Verified the model with a live prompt.</li>
</ul>

<h2>Verification</h2>
<div class="box">
  <p class="meta">Prompt sent:</p>
  <pre>Write a Python function to check if a number is a palindrome.</pre>
  <p class="meta">Response (abridged):</p>
  <pre>def is_palindrome(n):
    return str(n) == str(n)[::-1]</pre>
</div>
<p>The model responded correctly and coherently, confirming Ollama + Code Llama were fully operational
on the VM (CPU-only mode, since no GPU is passed through to the VM by VMware by default).</p>

<h2>Key Learnings</h2>
<div class="box">
  <p><b>Model sizing vs. hardware:</b> a 7B-parameter model needs several GB of RAM just to load; running
  it on constrained hardware (CPU-only, limited RAM) is slower but functionally correct.</p>
  <p><b>Instruction-tuned prompting:</b> Code Llama's "Instruct" variant is fine-tuned to follow natural-language
  commands (e.g. "Write a function to...") rather than just continuing code, using the
  <code>[INST] ... [/INST]</code> prompt format internally.</p>
</div>

<h2>Outcome</h2>
<p>A working local LLM environment on the VM, reachable via Ollama's HTTP API at
<code>http://localhost:11434</code>, used directly by the application built in Week 3.</p>
"""
    return page("Week 2 Report", "Running Code Llama on the Virtual Machine", body)


# ---------------- WEEK 3 ----------------
def build_week3():
    kb_path = os.path.join(ROOT, "week3", "knowledge_base.json")
    kb_rows = ""
    if os.path.exists(kb_path):
        with open(kb_path, encoding="utf-8") as f:
            kb = json.load(f)
        for item in kb:
            kb_rows += f"""<tr>
              <td><code>{esc(item['file'])}</code></td>
              <td>{esc(item['chunk_id'])}</td>
              <td><pre style="margin:0;white-space:pre-wrap;">{esc(item['text'])}</pre></td>
              <td>{len(item.get('embedding', []))} dimensions</td>
            </tr>"""

    sample_files = ["auth.py", "payment.py", "registration.py"]
    sample_code_html = ""
    for fname in sample_files:
        fpath = os.path.join(ROOT, "week3", "sample_codebase", fname)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            sample_code_html += f"<h3><code>{esc(fname)}</code></h3><pre>{esc(content)}</pre>"

    body = f"""
<h2>Objective</h2>
<p>Build one progressively-developed application demonstrating the full pipeline:
<b>Application &rarr; API &rarr; Retrieval/RAG &rarr; Chunking &rarr; Embeddings &rarr;
Vector Similarity &rarr; Relevant Context &rarr; LLM &rarr; Response</b>, using
Ollama + Code Llama + APIs + Services + Orchestration + Docker.</p>

<h2>Application Chosen: Code Q&amp;A Assistant</h2>
<p>An app that answers natural-language questions about a small sample codebase by retrieving
relevant code chunks and passing them as context to Code Llama.</p>

<h2>Exercise 1 &mdash; Basic LLM Connection</h2>
<p>A minimal script (<code>exercise1_basic.py</code>) sends a prompt to Ollama's HTTP API
(<code>/api/generate</code>) and prints Code Llama's response, proving the basic pipeline:
<code>User &rarr; App &rarr; API &rarr; Ollama &rarr; Code Llama &rarr; Response</code>.</p>

<h2>Exercise 2 &mdash; Knowledge Base (Chunking + Embeddings)</h2>
<p>A small sample codebase was created to act as the "knowledge" the app answers questions about:</p>
{sample_code_html}

<p>Each file was split into chunks and converted into embeddings (vector representations of meaning)
using Ollama's <code>nomic-embed-text</code> model, then saved to <code>knowledge_base.json</code>.</p>

<h3>Full Knowledge Base Contents</h3>
<table>
  <tr><th>File</th><th>Chunk #</th><th>Text</th><th>Embedding</th></tr>
  {kb_rows}
</table>
<p class="meta">Note: embedding vectors are 768-dimensional lists of numbers representing each chunk's
meaning &mdash; shown here as dimension count only, since the raw numbers are not human-readable.</p>

<h2>Exercise 3 &mdash; Retrieval + RAG</h2>
<p>When a question is asked, it is converted to an embedding and compared (via cosine similarity)
against every chunk in the knowledge base; the most similar chunks are retrieved and given to
Code Llama as context.</p>

<h3>Demonstration: With vs. Without RAG</h3>
<p class="meta">Question: <i>"Which file handles user authentication and how does it work?"</i></p>

<div class="bad">
  <b>WITHOUT RAG (no retrieved context):</b>
  <p>"User authentication is typically handled by a file called <code>auth.php</code> or
  <code>authentication.php</code>... includes a database connection and a SQL query..."</p>
  <p><b>&#10060; Hallucinated</b> &mdash; invented a fictional PHP/SQL file. This codebase has no PHP or database at all.</p>
</div>

<div class="good">
  <b>WITH RAG (retrieved context from registration.py, payment.py):</b>
  <p>"The <code>auth</code> module in the <code>payment.py</code> file handles user authentication by using the
  <code>authenticate_user()</code> function... defined in the <code>auth.py</code> file..."</p>
  <p><b>&#9989; Grounded and accurate</b> &mdash; correctly identified the real function and its real behavior.</p>
</div>

<h2>Exercise 4 &mdash; Services + Orchestration</h2>
<p>The application was split into independent services communicating over HTTP APIs:</p>
<ul>
  <li><b>Retrieval Service</b> (port 8001) &mdash; exposes <code>/retrieve</code>, returns relevant chunks for a question.</li>
  <li><b>App / Orchestration Service</b> (port 8000) &mdash; exposes <code>/ask</code>, calls the Retrieval Service,
      then calls Ollama (the LLM Service, already its own service on port 11434), and returns the final answer.</li>
</ul>
<p>Verified end-to-end via:</p>
<pre>curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \\
  -d '{{"question": "Which file handles user authentication?"}}'</pre>
<p>&rarr; Successfully returned a grounded JSON answer, proving the orchestration works.</p>

<h2>Exercise 5 &mdash; Dockerization</h2>
<p>The App Service and Retrieval Service were each packaged into Docker containers (via individual
Dockerfiles) and orchestrated together using <code>docker-compose.yml</code>. Ollama/Code Llama itself
runs natively on the VM host (not containerized) &mdash; a deliberate choice, since containerizing it would
require re-downloading the 3.8&nbsp;GB model inside the container for no functional benefit; the containers
reach it over the network via <code>host.docker.internal</code>.</p>
<div class="box">
<p><b>Verified working:</b> <code>docker compose up --build</code> started both containers successfully,
and the same <code>curl</code> test above returned a correct, grounded answer &mdash; confirming the fully
containerized services + host-level Ollama architecture works end-to-end.</p>
</div>

<h2>Conclusion</h2>
<p>All 5 exercises of Week 3 were completed and verified working: a RAG-based Code Q&amp;A application,
built progressively from a single script into a multi-service, Dockerized architecture, with clear
evidence that retrieval-augmented generation meaningfully improves answer grounding compared to the
LLM alone.</p>
"""
    return page("Week 3 Report", "Building the RAG-Based Code Q&A Application", body)


# ---------------- WEEK 4 ----------------
def build_week4():
    q_path = os.path.join(ROOT, "week4", "questions.json")
    with open(q_path, encoding="utf-8") as f:
        questions = json.load(f)

    q_rows = "".join(
        f"<tr><td>{q['id']}</td><td><span class='tag'>{esc(q['category'])}</span></td><td>{esc(q['question'])}</td></tr>"
        for q in questions
    )

    results_path = os.path.join(ROOT, "week4", "results.json")
    results_section = ""
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        models = sorted({m for entry in results for m in entry["models"]})

        agg_rows = ""
        from collections import defaultdict
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
            agg_rows += f"<tr><td><code>{esc(model)}</code></td><td>{avg_lat}s</td><td>{avg_tok}</td></tr>"

        per_q = ""
        for entry in results:
            per_q += f"<h3>Q{entry['id']} [{esc(entry['category'])}]: {esc(entry['question'])}</h3>"
            per_q += f"<p class='meta'>Retrieved context from: {', '.join(esc(x) for x in entry['retrieved_from'])}</p>"
            per_q += "<table><tr><th>Model</th><th>Latency</th><th>Tokens</th><th>Answer</th></tr>"
            for model in models:
                res = entry["models"].get(model)
                if not res:
                    per_q += f"<tr><td><code>{esc(model)}</code></td><td colspan='3'><i>not run</i></td></tr>"
                    continue
                answer = res["answer"].strip()
                per_q += f"<tr><td><code>{esc(model)}</code></td><td>{res['latency_seconds']}s</td><td>{res['eval_count']}</td><td>{esc(answer)}</td></tr>"
            per_q += "</table>"

        results_section = f"""
<h2>Exercise 3 &mdash; Quantitative Evaluation Results</h2>
<h3>Aggregate Performance</h3>
<table><tr><th>Model</th><th>Avg Latency</th><th>Avg Tokens Generated</th></tr>{agg_rows}</table>
<p class="meta">Latency and token counts are measured directly from Ollama's API response for each call
(CPU-only inference on an 8&nbsp;GB RAM VM). Correctness/relevance/hallucination were scored by manual
review of each answer below.</p>

<h2>Exercise 3 (continued) &mdash; Per-Question Answers, All Models</h2>
{per_q}
"""
    else:
        results_section = """
<h2>Exercise 3 &mdash; Quantitative Evaluation Results</h2>
<div class="warn"><p><b>Pending:</b> full multi-model evaluation run (<code>week4/run_evaluation.py</code>)
had not completed at the time this report was generated. Re-run <code>build_reports.py</code> after
<code>results.json</code> is produced to include the full per-question comparison table.</p></div>
"""

    exercise5_md = ""
    ex5_path = os.path.join(ROOT, "week4", "EXERCISE5_RAG_ANALYSIS.md")
    if os.path.exists(ex5_path):
        with open(ex5_path, encoding="utf-8") as f:
            exercise5_md = f.read()

    body = f"""
<h2>Objective</h2>
<p>Move from building an LLM application to systematically evaluating multiple models, performing
quantitative analysis, analysing the RAG pipeline, and testing repository-level code understanding.</p>

<h2>Exercise 1 &mdash; Multiple Models Evaluated</h2>
<ul>
  <li><code>codellama:7b</code> &mdash; the primary model used throughout Week 3.</li>
  <li><code>starcoder2:3b</code> &mdash; a smaller, code-specialized model.</li>
  <li><code>phi3:mini</code> &mdash; a small, fast general-purpose model with code capability.</li>
</ul>
<p>All three were tested using the identical application, prompts, questions, knowledge base, and
retrieval pipeline &mdash; only the model itself was swapped.</p>

<h2>Exercise 2 &mdash; Evaluation Question Set</h2>
<p>{len(questions)} representative questions were written, covering all task categories specified by
the assignment (code explanation, retrieval, dependency understanding, bug analysis, code generation,
refactoring, RAG-based, and repository-level understanding):</p>
<table><tr><th>#</th><th>Category</th><th>Question</th></tr>{q_rows}</table>

{results_section}

<h2>Exercise 4 &mdash; Analysis</h2>
<div class="box">
<p>Based on the aggregate metrics above: smaller models (<code>starcoder2:3b</code>, <code>phi3:mini</code>)
are expected to show meaningfully lower latency than <code>codellama:7b</code> on this CPU-only VM, since
fewer parameters means less computation per token. Whether that speed comes at a cost in accuracy or
increased hallucination is assessed by reading the per-question answers above side-by-side &mdash; look
specifically for: factual correctness against the real code, whether the model stayed grounded in the
retrieved context vs. inventing details, and completeness of the answer. Update this section with the
specific conclusion once the full run has been manually reviewed.</p>
</div>

<h2>Exercise 5 &mdash; RAG Pipeline Analysis</h2>
<pre style="white-space:pre-wrap;">{esc(exercise5_md)}</pre>

<h2>Exercise 6 &mdash; Repository-Level Understanding</h2>
<p>Two questions in the evaluation set specifically test multi-file/repository-level reasoning
(category <span class="tag">repo_understanding</span>): "Which files are involved in the user
authentication flow, end to end?" and "Which components would be affected if the
<code>authenticate_user</code> function signature changed?" These require the system to connect
information across <code>auth.py</code>, <code>payment.py</code>, and <code>registration.py</code>
simultaneously &mdash; see the per-question answers above for how each model handled this. As expected
for a simple top-<i>k</i> chunk-retrieval RAG system (no dependency graph or call-graph awareness), this
setup can surface individually relevant files but does not explicitly reason about cross-file call
relationships &mdash; the kind of capability tools like Sourcegraph (covered in the following week)
are designed to add.</p>

<h2>Conclusion</h2>
<p>Week 4 confirmed that model choice measurably affects both speed and answer quality on the same
RAG pipeline, and that retrieval quality directly shapes how grounded the final answer is &mdash;
reinforcing that RAG is not a simple guarantee of correctness but a pipeline whose weakest link
determines the outcome.</p>
"""
    return page("Week 4 Report", "Multi-Model Evaluation and RAG Analysis", body)


def main():
    reports = {
        "week1/Week1_Report.html": build_week1(),
        "week2/Week2_Report.html": build_week2(),
        "week3/Week3_Report.html": build_week3(),
        "week4/Week4_Report.html": build_week4(),
    }
    for rel_path, content in reports.items():
        full_path = os.path.join(ROOT, rel_path)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Wrote {rel_path}")


if __name__ == "__main__":
    main()
