"""
Agent router: determines which tool(s) to call for a given question.

Extends the simple keyword classifier in app.py with richer logic
for multi-tool hybrid queries.
"""
import re
from dataclasses import dataclass, field


@dataclass
class RoutingDecision:
    tools: list          # List of tool names to invoke: 'predict', 'query', 'ask'
    reasoning: str = ""  # Human-readable explanation of routing decision
    is_hybrid: bool = False


# Signals that strongly indicate each tool
_PREDICT_SIGNALS = [
    r"predict(ed)?\s+risk", r"risk\s+score", r"model\s+(score|output|predict)",
    r"forecast\s+risk", r"predicted\s+loss", r"ml\s+model",
]
_QUERY_SIGNALS = [
    r"\btop\b", r"\bhighest\b", r"\blowest\b", r"\btrend\b",
    r"\bcompare\b", r"\branking\b", r"\bhow\s+many\b", r"\btotal\b",
    r"\baverage\b", r"\blist\s+(all\s+)?counties\b", r"\blargest\s+increase\b",
    r"\byear.over.year\b", r"\bby\s+year\b",
]
_ASK_SIGNALS = [
    r"\bwhy\b", r"\bwhat\s+is\b", r"\bexplain\b", r"\bdescribe\b",
    r"\bhow\s+does\b", r"\bwhat\s+causes?\b", r"\bdocumentation\b",
    r"\breport\b", r"\bnarrative\b", r"\bhistory\s+of\b",
]


def _matches(question: str, patterns: list) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in patterns)


def route(question: str) -> RoutingDecision:
    """
    Determine which tools to invoke for the given question.

    Rules:
    - Predict-only:  question asks for model risk scores
    - Query-only:    question asks for structured analytics (rankings, trends)
    - Ask-only:      question asks for explanations or document-grounded reasoning
    - Hybrid (predict+query): asks for top counties by predicted risk + damage metrics
    - Hybrid (query+ask):     asks for analytics + contextual explanation
    """
    wants_predict = _matches(question, _PREDICT_SIGNALS)
    wants_query = _matches(question, _QUERY_SIGNALS)
    wants_ask = _matches(question, _ASK_SIGNALS)

    # Hybrid: explicitly asks for both structured data AND model prediction
    if wants_predict and wants_query:
        return RoutingDecision(
            tools=["predict", "query"],
            reasoning="Question requests ML risk scores combined with structured analytics",
            is_hybrid=True,
        )

    # Hybrid: structured result + explanation
    if wants_query and wants_ask:
        return RoutingDecision(
            tools=["query", "ask"],
            reasoning="Question requests analytics results plus contextual reasoning",
            is_hybrid=True,
        )

    if wants_predict:
        return RoutingDecision(
            tools=["predict"],
            reasoning="Question is primarily about ML risk prediction",
        )

    if wants_query:
        return RoutingDecision(
            tools=["query"],
            reasoning="Question is a structured analytics query over Gold data",
        )

    # Default: RAG ask for open-ended / explanatory questions
    return RoutingDecision(
        tools=["ask"],
        reasoning="Question is open-ended; routing to RAG retrieval + LLM reasoning",
    )
