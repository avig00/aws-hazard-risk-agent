"""
Prompt templates for the /ask RAG endpoint.
Constructs structured prompts for Bedrock Claude using retrieved context.
"""

SYSTEM_PROMPT = """You are a hazard risk intelligence analyst specializing in U.S. county-level
disaster risk assessment. You have access to FEMA disaster reports, NOAA event narratives,
and National Risk Index (NRI) documentation.

Your role is to answer questions about natural hazard risk, disaster history, and resilience
planning using only the provided source documents. Always:
- Ground your answers in the retrieved context
- Cite specific document sources when making claims
- Acknowledge when the context does not contain enough information to answer fully
- Avoid speculating beyond what the data supports"""


def build_ask_prompt(question: str, context_chunks: list) -> str:
    """
    Assemble the user message for the Bedrock Claude API call.

    Args:
        question: The user's natural-language question.
        context_chunks: List of retrieved chunks [{text, score, metadata}].

    Returns:
        Formatted prompt string combining context + question.
    """
    if not context_chunks:
        context_section = "No relevant documents were found in the knowledge base."
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown source")
            hazard = chunk.get("metadata", {}).get("hazard_type", "")
            header = f"[Source {i}: {source}"
            if hazard and hazard != "general":
                header += f" | Hazard: {hazard}"
            header += "]"
            context_parts.append(f"{header}\n{chunk['text']}")

        context_section = "\n\n---\n\n".join(context_parts)

    return f"""Based on the following retrieved documents, answer the question below.

=== RETRIEVED CONTEXT ===
{context_section}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
- Answer based only on the context provided above
- Cite the source numbers (e.g. "According to Source 2...") when referencing specific facts
- If the context is insufficient, say so clearly and explain what information is missing
- Be concise but complete"""


def build_citations(context_chunks: list) -> list:
    """
    Extract citation metadata from retrieved chunks for the API response.
    """
    citations = []
    seen = set()
    for chunk in context_chunks:
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")
        if source and source not in seen:
            seen.add(source)
            citations.append({
                "source": source,
                "hazard_type": meta.get("hazard_type", ""),
                "section": meta.get("section", ""),
                "score": chunk.get("score", 0.0),
            })
    return citations
