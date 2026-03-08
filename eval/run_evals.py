"""
Tier 3 — LLM-as-Judge Evaluation Harness
=========================================

Runs each test case from eval/test_cases.json against the live agent, then
asks Amazon Nova Lite to rate the response on three dimensions (1–5 scale):

  faithfulness  — Does the answer accurately reflect the data / retrieved docs?
  relevance     — Does the answer address the user's actual question?
  groundedness  — Are specific claims (numbers, names, tiers) traceable to the
                  tool output rather than hallucinated?

Usage
-----
  # Requires AWS credentials in environment or ~/.aws
  python eval/run_evals.py

  # Save results to JSON
  python eval/run_evals.py --output eval/results.json

  # Run a single test case by ID
  python eval/run_evals.py --id q2_flood_increase

Output
------
  Console: coloured summary table (pass ≥ 3.5 avg, warn 2.5–3.5, fail < 2.5)
  JSON: full scores + raw answers + judge explanations
"""

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import boto3

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
CASES_FILE = EVAL_DIR / "test_cases.json"
REPO_ROOT = EVAL_DIR.parent

sys.path.insert(0, str(REPO_ROOT))

# ── Score thresholds ───────────────────────────────────────────────────────────
PASS_THRESHOLD = 3.5   # avg ≥ 3.5 → PASS
WARN_THRESHOLD = 2.5   # avg ≥ 2.5 → WARN, < 2.5 → FAIL

# ── ANSI colours ──────────────────────────────────────────────────────────────
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


# ── Bedrock client ─────────────────────────────────────────────────────────────

def _make_bedrock_call():
    """Return a bedrock_call_fn wrapping Nova Lite."""
    client = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )

    def _call(system: str, user_message: str) -> str:
        resp = client.converse(
            modelId="us.amazon.nova-lite-v1:0",
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user_message}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.1},
        )
        return resp["output"]["message"]["content"][0]["text"]

    return _call


# ── Agent runner ──────────────────────────────────────────────────────────────

def _run_agent(question: str, bedrock_call_fn) -> dict[str, Any]:
    """Run the agent and return its full result dict."""
    from agent.orchestrator import run_agent
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    return run_agent(
        question=question,
        bedrock_call_fn=bedrock_call_fn,
        pinecone_api_key=pinecone_key,
    )


# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = textwrap.dedent("""\
    You are an expert evaluator for an AI assistant that answers questions about
    county-level hazard risk data.  You will be shown:

      USER QUESTION   — what the user asked
      TOOL OUTPUT     — structured data returned by the analytics/ML/RAG tool
      LLM ANSWER      — the AI assistant's final text response
      CRITERIA        — dimension-specific rubrics to apply

    Score each of the three dimensions below on a 1–5 integer scale:

      faithfulness  (1 = fabricates facts, 5 = fully accurate to tool output)
      relevance     (1 = off-topic, 5 = directly and completely addresses the question)
      groundedness  (1 = all numbers/names invented, 5 = every specific claim traceable to data)

    Return ONLY a JSON object with this exact structure (no prose, no markdown):
    {
      "faithfulness": <int 1-5>,
      "faithfulness_reason": "<one sentence>",
      "relevance": <int 1-5>,
      "relevance_reason": "<one sentence>",
      "groundedness": <int 1-5>,
      "groundedness_reason": "<one sentence>"
    }
""")


def _build_judge_user_message(
    question: str,
    tool_output_summary: str,
    answer: str,
    criteria: dict[str, str],
) -> str:
    criteria_text = "\n".join(
        f"  {dim}: {desc}" for dim, desc in criteria.items()
    )
    return textwrap.dedent(f"""\
        USER QUESTION:
        {question}

        TOOL OUTPUT SUMMARY:
        {tool_output_summary}

        LLM ANSWER:
        {answer}

        CRITERIA:
        {criteria_text}
    """)


def _summarise_tool_output(result: dict[str, Any]) -> str:
    """Produce a compact summary of what the tool returned for the judge."""
    lines = []
    lines.append(f"tools_used: {result.get('tool_used', [])}")

    query_out = result.get("tool_outputs", {}).get("query", {})
    if query_out:
        rows = query_out.get("results", [])
        lines.append(f"query rows returned: {len(rows)}")
        for i, row in enumerate(rows[:20], 1):
            lines.append(f"row {i}: {json.dumps(row)}")
        if len(rows) > 20:
            lines.append(f"... ({len(rows) - 20} more rows not shown)")
        intent = query_out.get("intent", "")
        if intent:
            lines.append(f"query intent: {intent}")
        sql = query_out.get("sql_executed", "")
        if sql:
            lines.append(f"sql (first 200 chars): {sql[:200]}")

    pred_out = result.get("tool_outputs", {}).get("predict", {})
    if pred_out and "risk_tier" in pred_out:
        lines.append(f"predicted risk_tier: {pred_out['risk_tier']}")
        lines.append(f"probabilities: {pred_out.get('probabilities', {})}")

    sources = result.get("sources", [])
    if sources:
        lines.append(f"rag sources ({len(sources)}): " +
                     ", ".join(s.get("source", "?") for s in sources[:3]))

    return "\n".join(lines)


# ── Scorer ────────────────────────────────────────────────────────────────────

def _judge_response(
    question: str,
    agent_result: dict[str, Any],
    criteria: dict[str, str],
    bedrock_call_fn,
) -> dict[str, Any]:
    """Ask Nova Lite to rate the agent response and return scores + reasons."""
    tool_summary = _summarise_tool_output(agent_result)
    answer = agent_result.get("answer", "")
    user_msg = _build_judge_user_message(question, tool_summary, answer, criteria)

    raw_json = bedrock_call_fn(_JUDGE_SYSTEM, user_msg)

    # Strip markdown code fences if present
    cleaned = raw_json.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        scores = json.loads(cleaned)
    except json.JSONDecodeError:
        scores = {
            "faithfulness": 0, "faithfulness_reason": f"Parse error: {cleaned[:80]}",
            "relevance": 0,     "relevance_reason": "",
            "groundedness": 0,  "groundedness_reason": "",
        }
    return scores


# ── Result rendering ──────────────────────────────────────────────────────────

def _colour_score(score: float) -> str:
    if score >= PASS_THRESHOLD:
        return f"{_GREEN}{score:.1f}{_RESET}"
    elif score >= WARN_THRESHOLD:
        return f"{_YELLOW}{score:.1f}{_RESET}"
    return f"{_RED}{score:.1f}{_RESET}"


def _verdict(avg: float) -> str:
    if avg >= PASS_THRESHOLD:
        return f"{_GREEN}{_BOLD}PASS{_RESET}"
    elif avg >= WARN_THRESHOLD:
        return f"{_YELLOW}{_BOLD}WARN{_RESET}"
    return f"{_RED}{_BOLD}FAIL{_RESET}"


def _print_result(case_id: str, question: str, scores: dict, answer: str) -> None:
    f = scores.get("faithfulness", 0)
    r = scores.get("relevance", 0)
    g = scores.get("groundedness", 0)
    avg = (f + r + g) / 3 if all(isinstance(x, (int, float)) for x in [f, r, g]) else 0
    print(f"\n{'─' * 72}")
    print(f"{_BOLD}{case_id}{_RESET}  {_verdict(avg)}  avg={_colour_score(avg)}")
    print(f"Q: {question[:80]}")
    print(f"  faithfulness={_colour_score(f)}  {scores.get('faithfulness_reason', '')[:80]}")
    print(f"  relevance   ={_colour_score(r)}  {scores.get('relevance_reason', '')[:80]}")
    print(f"  groundedness={_colour_score(g)}  {scores.get('groundedness_reason', '')[:80]}")
    print(f"  Answer ({len(answer)} chars): {answer[:120]}{'…' if len(answer) > 120 else ''}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_summary(all_results: list[dict], cases_by_id: dict) -> None:
    """Print a comprehensive summary with per-tool and per-dimension breakdowns."""
    scored = [r for r in all_results if "scores" in r]
    if not scored:
        return

    avgs = [r["avg_score"] for r in scored]
    overall = sum(avgs) / len(avgs)
    passes = sum(1 for a in avgs if a >= PASS_THRESHOLD)
    warns  = sum(1 for a in avgs if WARN_THRESHOLD <= a < PASS_THRESHOLD)
    fails  = sum(1 for a in avgs if a < WARN_THRESHOLD)

    # ── Overall summary ───────────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print(f"{_BOLD}SUMMARY{_RESET}  overall avg={_colour_score(overall)}  "
          f"{_GREEN}PASS{_RESET}={passes}  {_YELLOW}WARN{_RESET}={warns}  "
          f"{_RED}FAIL{_RESET}={fails}  ({len(all_results)} cases)")

    # ── Per-dimension averages ────────────────────────────────────────────────
    dims = ["faithfulness", "relevance", "groundedness"]
    dim_avgs = {}
    for d in dims:
        vals = [r["scores"][d] for r in scored if isinstance(r["scores"].get(d), (int, float))]
        dim_avgs[d] = sum(vals) / len(vals) if vals else 0.0

    print(f"\n{_BOLD}Dimension averages:{_RESET}")
    for d, avg in dim_avgs.items():
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"  {d:<14} {bar}  {_colour_score(avg)}/5")

    # ── Per-tool breakdown ────────────────────────────────────────────────────
    tool_groups: dict[str, list[dict]] = {}
    for r in scored:
        tool = cases_by_id.get(r["id"], {}).get("tool", "unknown")
        tool_groups.setdefault(tool, []).append(r)

    if len(tool_groups) > 1:
        print(f"\n{_BOLD}Per-tool breakdown:{_RESET}")
        print(f"  {'Tool':<10} {'Cases':>5}  {'Avg':>5}  {'Pass':>4}  {'Warn':>4}  {'Fail':>4}")
        print(f"  {'─'*10} {'─'*5}  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*4}")
        for tool_name, rows in sorted(tool_groups.items()):
            t_avgs = [r["avg_score"] for r in rows]
            t_overall = sum(t_avgs) / len(t_avgs)
            t_pass = sum(1 for a in t_avgs if a >= PASS_THRESHOLD)
            t_warn = sum(1 for a in t_avgs if WARN_THRESHOLD <= a < PASS_THRESHOLD)
            t_fail = sum(1 for a in t_avgs if a < WARN_THRESHOLD)
            print(f"  {tool_name:<10} {len(rows):>5}  {_colour_score(t_overall):>5}  "
                  f"{t_pass:>4}  {t_warn:>4}  {t_fail:>4}")

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{_BOLD}Results table:{_RESET}")
    print(f"  {'ID':<24} {'Tool':<8} {'Faith':>5}  {'Relev':>5}  {'Grnd':>5}  {'Avg':>5}  Verdict")
    print(f"  {'─'*24} {'─'*8} {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*7}")
    for r in scored:
        tool = cases_by_id.get(r["id"], {}).get("tool", "?")
        s = r["scores"]
        f_s = s.get("faithfulness", 0)
        r_s = s.get("relevance", 0)
        g_s = s.get("groundedness", 0)
        a = r["avg_score"]
        v = "PASS" if a >= PASS_THRESHOLD else ("WARN" if a >= WARN_THRESHOLD else "FAIL")
        v_col = (_GREEN if a >= PASS_THRESHOLD else (_YELLOW if a >= WARN_THRESHOLD else _RED))
        print(f"  {r['id']:<24} {tool:<8} {_colour_score(f_s):>5}  {_colour_score(r_s):>5}  "
              f"{_colour_score(g_s):>5}  {_colour_score(a):>5}  {v_col}{v}{_RESET}")

    if fails > 0:
        print(f"\n{_RED}{_BOLD}Failures requiring attention:{_RESET}")
        for r in scored:
            if r["avg_score"] < WARN_THRESHOLD:
                print(f"  • {r['id']}: avg={r['avg_score']:.2f}  Q: {r['question'][:60]}")

    print(f"{'═' * 72}")


def run_evals(case_ids: list[str] | None = None) -> list[dict]:
    cases = json.loads(CASES_FILE.read_text())
    cases_by_id = {c["id"]: c for c in cases}
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    if not cases:
        print("No matching test cases found.")
        sys.exit(1)

    bedrock_call_fn = _make_bedrock_call()
    all_results = []

    print(f"\n{_BOLD}Tier 3 LLM-as-Judge Evaluation — {len(cases)} cases{_RESET}")
    print(f"Model: us.amazon.nova-lite-v1:0  |  Pass ≥{PASS_THRESHOLD}  Warn ≥{WARN_THRESHOLD}")

    for case in cases:
        case_id = case["id"]
        question = case["question"]
        criteria = case["criteria"]

        print(f"\n[{case_id}] Running agent for: {question[:60]}…", end="", flush=True)
        t0 = time.time()
        try:
            agent_result = _run_agent(question, bedrock_call_fn)
        except Exception as exc:
            print(f"\n  ERROR running agent: {exc}")
            all_results.append({"id": case_id, "error": str(exc)})
            continue
        elapsed_agent = time.time() - t0

        print(f" ({elapsed_agent:.1f}s) → judging…", end="", flush=True)
        t1 = time.time()
        try:
            scores = _judge_response(question, agent_result, criteria, bedrock_call_fn)
        except Exception as exc:
            print(f"\n  ERROR in judge: {exc}")
            all_results.append({"id": case_id, "error": f"judge: {exc}"})
            continue
        elapsed_judge = time.time() - t1

        answer = agent_result.get("answer", "")
        _print_result(case_id, question, scores, answer)
        print(f"  (agent {elapsed_agent:.1f}s, judge {elapsed_judge:.1f}s)")

        all_results.append({
            "id": case_id,
            "tool": case.get("tool", "unknown"),
            "question": question,
            "answer": answer,
            "tool_used": agent_result.get("tool_used", []),
            "scores": scores,
            "avg_score": round(
                (scores.get("faithfulness", 0) + scores.get("relevance", 0) +
                 scores.get("groundedness", 0)) / 3, 2
            ),
            "elapsed_agent_s": round(elapsed_agent, 2),
            "elapsed_judge_s": round(elapsed_judge, 2),
        })

    _print_summary(all_results, cases_by_id)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-judge evals")
    parser.add_argument(
        "--output",
        default=str(EVAL_DIR / "results.json"),
        help="Path to save JSON results (default: eval/results.json)",
    )
    parser.add_argument("--id", nargs="+", dest="ids", help="Run specific case IDs only")
    args = parser.parse_args()

    results = run_evals(case_ids=args.ids)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
