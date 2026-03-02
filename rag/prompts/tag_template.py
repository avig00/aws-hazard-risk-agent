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
- Reference specific values from the table (county names, scores, years, percentages)
- Use your domain knowledge to add context (e.g. why a county ranks highly, what drives NRI scores)
- Highlight the top findings — do not just re-state the table rows
- Note any important caveats: data completeness, time period scope, what the metric measures
- Be concise but analytically rigorous — write like a data analyst presenting to a risk team"""


# Column descriptions injected into the prompt so Claude understands the schema
_COLUMN_DESCRIPTIONS = {
    "county_fips":              "5-digit FIPS code identifying the county",
    "county_name":              "County name",
    "state":                    "U.S. state abbreviation",
    "year":                     "Calendar year",
    "avg_expected_loss":        "Average NRI Expected Annual Loss ($) — the primary risk metric",
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
    "avg_events_early":         "Average annual events in the earlier period",
    "avg_events_recent":        "Average annual events in the more recent period",
    "absolute_increase":        "Absolute change in average events between periods",
    "pct_increase":             "Percentage change in average events between periods (%)",
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

    return f"""QUESTION: {question}

QUERY EXECUTED (governed SQL — Gold layer only):
```sql
{sql_executed.strip()}
```

RESULTS ({row_count} rows total):
{data_note_section}{column_section}
{table_text}

---
Please provide:
1. A direct, specific answer to the question referencing actual values from the results
2. The top 2–3 notable findings or patterns in the data
3. Any relevant context from your domain knowledge (what drives these scores, known risk factors)
4. A brief caveat if the data has limitations relevant to this question"""
