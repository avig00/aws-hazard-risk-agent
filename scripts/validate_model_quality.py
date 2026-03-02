"""
Quality validation: Amazon Nova Lite vs Llama 3.3 70B (Claude 3 Haiku proxy)
on real project prompts (TAG synthesis + RAG synthesis).

Uses actual Athena query results and real Pinecone-retrieved chunks.
Scores both models on 5 dimensions, prints side-by-side for comparison.
"""
import json
import os
import sys
import textwrap
import time
from pathlib import Path

import boto3

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rag.prompts.ask_template import SYSTEM_PROMPT, build_ask_prompt
from rag.prompts.tag_template import TAG_SYSTEM_PROMPT, build_tag_prompt
from analytics.query_engine import run_query
from rag.retrieval.retrieve import retrieve_similar

# ── config ────────────────────────────────────────────────────────────────────
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")

MODELS = {
    "nova-lite": {
        "id": "us.amazon.nova-lite-v1:0",
        "label": "Amazon Nova Lite",
        "cost_in": 0.060,
        "cost_out": 0.240,
    },
    "llama-70b": {
        "id": "us.meta.llama3-3-70b-instruct-v1:0",
        "label": "Llama 3.3 70B (Haiku quality proxy)",
        "cost_in": 0.720,
        "cost_out": 0.720,
    },
}

DIVIDER = "─" * 80
SEPARATOR = "═" * 80

# ── Bedrock Converse helper ───────────────────────────────────────────────────
def converse(model_id: str, system_text: str, user_text: str, max_tokens: int = 1024) -> tuple[str, int, int]:
    """Call Bedrock Converse API. Returns (text, input_tokens, output_tokens)."""
    client = boto3.client("bedrock-runtime", region_name=REGION)
    resp = client.converse(
        modelId=model_id,
        system=[{"text": system_text}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.1},
    )
    text = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    return text, usage.get("inputTokens", 0), usage.get("outputTokens", 0)


# ── Scoring helpers ───────────────────────────────────────────────────────────
DOMAIN_TERMS = [
    "NRI", "Expected Annual Loss", "EAL", "FEMA", "NOAA", "exposure",
    "vulnerability", "resilience", "storm surge", "NFIP", "risk index",
    "social vulnerability", "SVI", "county", "flood zone", "FIPS",
]

def score_response(text: str, reference_values: list[str]) -> dict:
    """
    Score a model response on 5 dimensions (0–2 each, max 10).
    - domain_vocab: Uses hazard-risk domain terms correctly
    - data_citation: Cites specific numbers/county names from query results
    - analytical_depth: Goes beyond restating data; adds context or patterns
    - appropriate_caveats: Notes data limitations or scope
    - format_compliance: Follows the requested output structure
    """
    text_lower = text.lower()

    # 1. Domain vocabulary (0-2)
    term_hits = sum(1 for t in DOMAIN_TERMS if t.lower() in text_lower)
    domain_vocab = min(2, term_hits // 2)

    # 2. Data citation — checks that specific values from results appear
    cited = sum(1 for v in reference_values if v.lower() in text_lower)
    data_citation = 2 if cited >= len(reference_values) * 0.6 else (1 if cited > 0 else 0)

    # 3. Analytical depth — presence of interpretive language
    interpretive_phrases = [
        "suggests", "indicates", "driven by", "explains", "pattern",
        "notable", "likely", "because", "reflects", "due to", "top finding",
        "key insight", "worth noting", "significant", "higher than", "compared to",
    ]
    depth_hits = sum(1 for p in interpretive_phrases if p in text_lower)
    analytical_depth = min(2, depth_hits // 2)

    # 4. Caveats — does the model acknowledge limitations?
    caveat_phrases = [
        "caveat", "limitation", "note that", "keep in mind", "important",
        "does not include", "may not", "incomplete", "period", "only covers",
        "time period", "scope", "not account", "should consider",
    ]
    caveat_hit = any(p in text_lower for p in caveat_phrases)
    appropriate_caveats = 2 if caveat_hit else 0

    # 5. Format compliance — structured response with numbered/bulleted points
    has_structure = (
        text.count("1.") > 0 or
        text.count("•") > 0 or
        text.count("-") > 2 or
        text.count("\n\n") > 1
    )
    format_compliance = 2 if has_structure else 1

    total = domain_vocab + data_citation + analytical_depth + appropriate_caveats + format_compliance
    return {
        "domain_vocab": domain_vocab,
        "data_citation": data_citation,
        "analytical_depth": analytical_depth,
        "appropriate_caveats": appropriate_caveats,
        "format_compliance": format_compliance,
        "total": total,
    }


def print_comparison(test_name: str, results: dict, reference_values: list[str]):
    """Print side-by-side scores and truncated outputs."""
    print(f"\n{SEPARATOR}")
    print(f"TEST: {test_name}")
    print(SEPARATOR)

    scores = {}
    for model_key, data in results.items():
        label = MODELS[model_key]["label"]
        text = data["text"]
        s = score_response(text, reference_values)
        scores[model_key] = s
        latency = data["latency_s"]
        in_tok = data["input_tokens"]
        out_tok = data["output_tokens"]

        print(f"\n{'─'*40}")
        print(f"  {label}")
        print(f"  Latency: {latency:.1f}s  |  Tokens: {in_tok} in / {out_tok} out")
        print(f"  Score: {s['total']}/10  "
              f"(vocab={s['domain_vocab']} citation={s['data_citation']} "
              f"depth={s['analytical_depth']} caveat={s['appropriate_caveats']} "
              f"format={s['format_compliance']})")
        print(f"{'─'*40}")
        # Print first 800 chars of response
        preview = textwrap.fill(text[:800], width=76, subsequent_indent="  ")
        print(f"  {preview}")
        if len(text) > 800:
            print(f"  ... [{len(text)-800} more chars]")

    # Verdict
    nova_score = scores.get("nova-lite", {}).get("total", 0)
    llama_score = scores.get("llama-70b", {}).get("total", 0)
    delta = llama_score - nova_score
    print(f"\n  VERDICT: Nova Lite {nova_score}/10 vs Llama 70B {llama_score}/10"
          f"  (delta = {delta:+d})")
    if delta <= 1:
        print("  → Quality difference is NEGLIGIBLE (≤1 point)")
    elif delta == 2:
        print("  → Quality difference is MINOR (2 points)")
    else:
        print("  → Quality difference is SIGNIFICANT (>2 points)")


# ── Test 1: TAG synthesis ─────────────────────────────────────────────────────
def test_tag_synthesis():
    print(f"\n{SEPARATOR}")
    print("FETCHING real Athena data for TAG tests...")
    print(SEPARATOR)

    tag_tests = [
        {
            "name": "TAG-1: Top counties by flood risk (NRI Expected Loss)",
            "question": "Which 10 US counties have the highest flood expected annual loss?",
            "query": "top 10 counties by flood risk with expected loss NRI score",
        },
        {
            "name": "TAG-2: Hurricane trend 2015–2022",
            "question": "Has hurricane damage increased from 2015 to 2022?",
            "query": "hurricane damage trend from 2015 to 2022",
        },
    ]

    for test in tag_tests:
        print(f"\nRunning Athena query: {test['query'][:60]}...")
        try:
            qr = run_query(test["query"], limit=10)
            rows = qr.get("results", [])
            sql = qr.get("sql_executed", "")
            intent = qr.get("intent", "")
            row_count = len(rows)
            print(f"  → {row_count} rows returned (intent={intent})")
        except Exception as e:
            print(f"  ✗ Athena query failed: {e}")
            continue

        if not rows:
            print("  ✗ No rows — skipping this test")
            continue

        # Build TAG prompt (same as production)
        user_msg = build_tag_prompt(
            question=test["question"],
            results=rows,
            sql_executed=sql,
            intent=intent,
            row_count=row_count,
        )

        # Extract reference values for scoring
        ref_vals = []
        for row in rows[:5]:
            for v in row.values():
                if v and str(v).strip():
                    ref_vals.append(str(v))
        ref_vals = ref_vals[:12]

        model_results = {}
        for model_key, model_cfg in MODELS.items():
            print(f"  Calling {model_cfg['label']}...")
            try:
                t0 = time.time()
                text, in_tok, out_tok = converse(
                    model_id=model_cfg["id"],
                    system_text=TAG_SYSTEM_PROMPT,
                    user_text=user_msg,
                )
                latency = time.time() - t0
                model_results[model_key] = {
                    "text": text,
                    "latency_s": latency,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
                print(f"    ✓ {latency:.1f}s, {out_tok} output tokens")
            except Exception as e:
                print(f"    ✗ {model_cfg['label']} failed: {e}")
                model_results[model_key] = {
                    "text": f"[ERROR: {e}]",
                    "latency_s": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

        print_comparison(test["name"], model_results, ref_vals)


# ── Test 2: RAG synthesis ─────────────────────────────────────────────────────
def test_rag_synthesis():
    print(f"\n{SEPARATOR}")
    print("FETCHING real Pinecone chunks for RAG tests...")
    print(SEPARATOR)

    rag_tests = [
        {
            "name": "RAG-1: Coastal hurricane vulnerability",
            "question": "Why are coastal counties more vulnerable to hurricanes and what factors drive their risk scores?",
        },
        {
            "name": "RAG-2: Social vulnerability and disaster recovery",
            "question": "How does social vulnerability affect disaster recovery outcomes in low-income counties?",
        },
        {
            "name": "RAG-3: NRI Expected Loss methodology",
            "question": "How is the FEMA National Risk Index Expected Annual Loss calculated and what hazards does it include?",
        },
    ]

    for test in rag_tests:
        print(f"\nRetrieving Pinecone chunks: '{test['question'][:60]}...'")
        try:
            chunks = retrieve_similar(
                question=test["question"],
                k=4,
                pinecone_api_key=PINECONE_API_KEY,
            )
            print(f"  → {len(chunks)} chunks retrieved, top score={chunks[0]['score']:.3f}" if chunks else "  → 0 chunks")
        except Exception as e:
            print(f"  ✗ Pinecone retrieval failed: {e}")
            continue

        if not chunks:
            print("  ✗ No chunks — skipping this test")
            continue

        # Build RAG prompt (same as production)
        user_msg = build_ask_prompt(question=test["question"], context_chunks=chunks)

        # Reference values: first lines of each chunk
        ref_vals = []
        for c in chunks:
            words = c["text"].split()[:8]
            ref_vals.extend(words)
        ref_vals = [w.strip(".,()") for w in ref_vals if len(w) > 4][:15]

        model_results = {}
        for model_key, model_cfg in MODELS.items():
            print(f"  Calling {model_cfg['label']}...")
            try:
                t0 = time.time()
                text, in_tok, out_tok = converse(
                    model_id=model_cfg["id"],
                    system_text=SYSTEM_PROMPT,
                    user_text=user_msg,
                )
                latency = time.time() - t0
                model_results[model_key] = {
                    "text": text,
                    "latency_s": latency,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                }
                print(f"    ✓ {latency:.1f}s, {out_tok} output tokens")
            except Exception as e:
                print(f"    ✗ {model_cfg['label']} failed: {e}")
                model_results[model_key] = {
                    "text": f"[ERROR: {e}]",
                    "latency_s": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }

        print_comparison(test["name"], model_results, ref_vals)


# ── Summary ───────────────────────────────────────────────────────────────────
def print_final_verdict():
    print(f"\n{SEPARATOR}")
    print("FINAL VERDICT SUMMARY")
    print(SEPARATOR)
    print("""
Amazon Nova Lite ($0.06/$0.24 per 1M) vs Llama 3.3 70B ($0.72/$0.72 per 1M)
Llama 3.3 70B serves as quality CEILING PROXY for Claude 3 Haiku ($0.25/$1.25 per 1M).

Interpretation guide:
  - If Nova Lite scores within 2 pts of Llama 70B across all tests:
    → Switch to Nova Lite; quality trade-off is negligible for this project.
  - If Nova Lite scores 3+ pts lower on TAG tests specifically:
    → TAG synthesis (structured data narrative) benefits from stronger model.
  - If Nova Lite scores 3+ pts lower on RAG tests:
    → RAG grounding (doc Q&A) needs better model; consider Nova Pro or Llama 70B.
""")


if __name__ == "__main__":
    print(SEPARATOR)
    print("AWS Hazard Risk Agent — LLM Quality Validation")
    print("Comparing: Amazon Nova Lite vs Llama 3.3 70B (Claude 3 Haiku proxy)")
    print(SEPARATOR)

    test_tag_synthesis()
    test_rag_synthesis()
    print_final_verdict()
