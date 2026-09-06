# Week 4, Exercise 5 — RAG Pipeline Analysis

This exercise traces `Question -> Retrieved Context -> LLM Response` for selected questions, to understand how retrieval quality affects the final answer.

## Example 1: "Which file handles user authentication and how does it work?"

**Without RAG (no retrieved context, model relies only on its training data):**

> User authentication is typically handled by a file called `auth.php` or `authentication.php`. This file is responsible for verifying the user's credentials... The file typically includes a database connection and a SQL query that retrieves the user's information from the database...

**Analysis:** The model **hallucinated completely**. It invented a fictional `auth.php` file using PHP and SQL — technologies that don't exist anywhere in this codebase, which is pure Python with no database. This is a clean example of an LLM falling back on generic patterns from its training data when it has no real information about the actual system being asked about.

**With RAG (retrieved: `registration.py`, `payment.py`):**

> The `auth` module in the `payment.py` file handles user authentication by using the `authenticate_user()` function. This function takes a username, password, and a dictionary of users as input, and returns a boolean value... The `authenticate_user()` function is defined in the `auth.py` file, which is imported in the `payment.py` file...

**Analysis:** With retrieved context, the model correctly identified the real function (`authenticate_user()`) and described its real behavior accurately. This is a large, clear improvement over the hallucinated answer.

**However — a retrieval quality note:** the top-2 retrieved chunks were from `registration.py` and `payment.py`, not `auth.py` itself (where `authenticate_user` is actually *defined*). The model still answered correctly because `payment.py`'s chunk includes the `import authenticate_user` line and references the function, so enough information leaked through — but this shows the retrieval step picked *related* chunks rather than the *most relevant* one. With a larger/more ambiguous codebase, this kind of near-miss retrieval could cause an incomplete or slightly wrong answer.

## Retrieval -> Context -> Response relationship: what this shows

| Stage | What happened | Effect |
|---|---|---|
| Retrieval | Picked `registration.py` + `payment.py` over `auth.py` | "Good enough" — related, not exact |
| Context | Contained a reference to `authenticate_user()` via the import line | Gave the model just enough signal |
| Response | Correct, grounded answer | RAG succeeded despite imperfect retrieval |

**Conclusion:** This one example demonstrates the core lesson of the exercise: RAG is not a simple guarantee of correctness just because *some* context was retrieved — the *relevance* of what's retrieved directly shapes response quality. Here, retrieval was imperfect (top match wasn't the defining file) but still adequate. In a case where retrieval pulled in fully irrelevant chunks (e.g. `registration.py` alone, without the `payment.py` chunk that had the import reference), the model would likely have reverted to a hallucinated or incomplete answer, similar to the no-RAG case above.

*(Additional examples from the 12-question Week 4 evaluation set are included in `REPORT.md`, generated after running `run_evaluation.py`.)*
