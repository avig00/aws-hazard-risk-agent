"""
Prompt templates for Table Augmented Generation (TAG).

TAG is distinct from RAG: instead of grounding answers in retrieved text chunks,
the LLM receives structured query results from Athena and synthesizes a
data-backed narrative — combining the computed metrics with world knowledge
about hazard risk to produce analyst-quality answers.

The TAG prompt is intentionally different from ask_template.py:
  - RAG: "answer only from these retrieved documents"
  - TAG: "interpret these query results and add analytical context"
"""

TAG_SYSTEM_PROMPT = """You are a senior hazard risk data analyst with deep expertise in U.S. \
county-level disaster risk, FEMA claims data, NOAA event records, and the National Risk Index (NRI).

You have been given the results of a governed SQL query executed over the FEMA/NOAA Gold-layer \
analytics database. Your role is to interpret the numbers, surface key patterns, and provide \
an authoritative, data-backed answer to the user's question.

Guidelines:
- ONLY cite county names, state names, figures, percentages, and scores that appear literally in the RESULTS table below. Do not generate additional examples, infer values for unlisted counties, or add specific claims from outside this dataset.
- Use your domain knowledge solely to explain WHY the data shows what it shows — not to introduce additional data points.
- Highlight the top findings — do not just re-state the table rows
- Note any important caveats: data completeness, time period scope, what the metric measures
- Be concise but analytically rigorous — write like a data analyst presenting to a risk team"""


# Column descriptions injected into the prompt so Claude understands the schema
_COLUMN_DESCRIPTIONS = {
    "county_fips":              "5-digit FIPS code identifying the county",
    "county_name":              "County name",
    "state":                    "U.S. state abbreviation",
    "year":                     "Calendar year",
    "avg_eal_score":            "Average NRI Expected Annual Loss index score (0–100) — higher = greater expected loss relative to other counties; NOT a dollar amount",
    "avg_expected_loss":        "Average NRI Expected Annual Loss index score (0–100) — NOT a dollar amount",
    "NRI_ExpectedLoss":         "NRI Expected Annual Loss ($) — estimated average annual dollar loss from natural hazards",
    "avg_exposure":             "Average NRI Exposure score (0–1) — built environment and population at risk",
    "NRI_Exposure":             "NRI Exposure score (0–1)",
    "avg_vulnerability":        "Average NRI Social Vulnerability score (0–1) — community sensitivity to hazards",
    "NRI_SocialVulnerability":  "NRI Social Vulnerability (0–1) — income, race, disability, housing factors",
    "avg_resilience":           "Average NRI Community Resilience (0–1) — capacity to recover",
    "NRI_CommunityResilience":  "NRI Community Resilience (0–1)",
    "total_events":             "Total number of hazard events recorded (NOAA Storm Events database)",
    "total_fatalities":         "Total direct fatalities from all hazard events",
    "fema_claim_count":         "Number of FEMA individual assistance claims filed",
    "fema_property_damage":     "Total FEMA-reported property damage ($)",
    "fema_total_assistance":    "Total FEMA individual + public assistance disbursed ($)",
    "avg_property_damage":      "Average FEMA-reported property damage per year ($)",
    "total_claims":             "Total FEMA claims filed across the period",
    "counties_affected":        "Number of distinct counties with recorded events",
    "years_on_record":          "Number of years with data available for this county",
    "total_declarations":       "Total FEMA disaster declarations summed across all counties in the state",
    "avg_declarations_per_county": "Average FEMA declarations per county in the state",
    "total_fema_damage":        "Total FEMA-reported damage across the state ($)",
    "avg_events_early":         "Average annual events in the earlier period",
    "avg_events_recent":        "Average annual events in the more recent period",
    "events_early_period":      "Total events in the earlier time period (per-hazard count)",
    "events_recent_period":     "Total events in the more recent time period (per-hazard count)",
    "absolute_increase":        "Absolute change in events between the two periods",
    "pct_increase":             "Percentage change in events between the two periods (%)",
    "property_damage":          "Reported property damage ($)",
    "expected_loss":            "NRI Expected Annual Loss ($)",
    "exposure":                 "NRI Exposure score",
    "vulnerability":            "NRI Social Vulnerability score",
    "resilience":               "NRI Community Resilience score",
}


def _describe_columns(row_keys: list) -> str:
    """Build a column legend for columns present in the result set."""
    lines = []
    for col in row_keys:
        desc = _COLUMN_DESCRIPTIONS.get(col)
        if desc:
            lines.append(f"  - {col}: {desc}")
    return "\n".join(lines) if lines else ""


def _format_results_table(rows: list, max_rows: int = 25) -> str:
    """
    Format query results as a plain-text table for the LLM.
    Truncates to max_rows to stay within token limits.
    """
    if not rows:
        return "No rows returned."

    display_rows = rows[:max_rows]
    headers = list(display_rows[0].keys())

    # Header row
    lines = [" | ".join(str(h) for h in headers)]
    lines.append("-" * len(lines[0]))

    for row in display_rows:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))

    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more rows truncated)")

    return "\n".join(lines)


def build_tag_prompt(
    question: str,
    results: list,
    sql_executed: str,
    intent: str,
    row_count: int,
    data_note: str = "",
    order_col: str = "",
) -> str:
    """
    Assemble the TAG user message for Bedrock Claude.

    Args:
        question:     The user's original natural-language question.
        results:      List of dicts from Athena (query results).
        sql_executed: The compiled SQL that was run — shown to Claude for transparency.
        intent:       The matched SQL template name (e.g. 'top_counties_by_risk').
        row_count:    Total rows returned (may exceed what's shown if truncated).
        data_note:    Optional plain-English note about data limitations detected
                      upstream (e.g. hazard not filterable, all-zero columns).
                      When present it is injected before the results so the LLM
                      can relay it to the user instead of producing a misleading answer.

    Returns:
        Formatted prompt string ready to send to Bedrock.
    """
    table_text = _format_results_table(results)
    col_legend = _describe_columns(list(results[0].keys()) if results else [])

    column_section = (
        f"\nCOLUMN DEFINITIONS:\n{col_legend}\n" if col_legend else ""
    )

    data_note_section = (
        f"\n⚠️  DATA LIMITATION NOTE (must relay to user):\n{data_note}\n"
        if data_note else ""
    )

    order_col_note = (
        f"\nRANKING NOTE: Results are sorted by {order_col}. "
        f"Do NOT make comparative claims about other columns in the result (e.g., 'also has the most fatalities', "
        f"'also leads in injuries') — those columns are included as context only and are NOT globally ranked. "
        f"A county that ranks high on {order_col} may not rank highest on other metrics across the full dataset, "
        f"because this result set is limited to the top rows by {order_col} only.\n"
    ) if order_col else ""

    return f"""QUESTION: {question}

QUERY EXECUTED (governed SQL — Gold layer only):
```sql
{sql_executed.strip()}
```

RESULTS ({row_count} rows total):
{data_note_section}{order_col_note}{column_section}
{table_text}

---
Write your response as a senior data analyst presenting to an executive risk team.
- Open directly with 1–2 sentences citing specific values from the data (county names, dollar figures, percentages, event counts).
- Follow with 2–3 concise bullet points for the key patterns or comparisons you observe.
- Close with one sentence of analytical context or caveat only if the data has a meaningful limitation.
Do NOT use generic section headers like "Answer to the Question:", "Top Findings:", "Context:", or "Caveat:" — write in direct analyst prose with bullets only for the findings list.
Do NOT wrap numbers, dollar amounts, percentages, or county names in backticks or code formatting.
IMPORTANT: The RESULTS table is already sorted in the order that answers the question — row 1 is the highest-ranked result. Always cite row 1 first. Do not re-rank or reorder the data in your narrative. If the question asks "most", "highest", or "largest", the answer is ALWAYS the entity in Row 1 — do NOT substitute a per-capita, per-county, average, or normalized metric as the ranking criterion unless the question explicitly asks for a rate or average.
IMPORTANT: If the data contradicts the user's question (e.g., the user asked about "increases" but all values in the data show decreases or zero change), state this directly and prominently in the first sentence. Do not describe opposite findings as if they partially answer the question.
IMPORTANT: If multiple rows share the same value for the ranking metric, describe them as tied — do NOT say one county "follows closely" or "comes in second" when the values are identical. Example: "Cumberland County and Gates County are tied with 1 event each."
IMPORTANT: Do NOT make geographic summary claims (e.g., "all counties are in California", "dominated by Illinois", "concentrated in the South") unless you can verify the claim against every row in the table shown above. Count the states in the table before making any such claim. Never use outside knowledge about which states tend to be high-risk — base geographic observations solely on the state values present in the RESULTS rows."""
