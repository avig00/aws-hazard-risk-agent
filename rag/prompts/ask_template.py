"""
Prompt templates for the /ask RAG endpoint.
Constructs structured prompts for Bedrock Claude using retrieved context.
"""

SYSTEM_PROMPT = """You are a hazard risk intelligence analyst specializing in U.S. county-level
disaster risk assessment. You have deep expertise in FEMA disaster programs, NOAA storm data,
the National Risk Index (NRI) methodology, and natural hazard science.

Your role is to answer questions about natural hazard risk, disaster history, and resilience
planning. Follow these priorities:
1. When retrieved context is available, ground your answer in those documents and cite sources
2. When context is absent or insufficient, draw on your expert knowledge of U.S. hazard risk,
   NRI methodology, FEMA programs, and NOAA hazard science to give a complete, useful answer
3. Clearly distinguish retrieved facts (cite source numbers) from general domain knowledge
4. Never refuse to answer solely because retrieved context is limited — use your expertise"""


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

    return f"""Answer the question below using the retrieved context and your domain expertise.

=== RETRIEVED CONTEXT ===
{context_section}

=== QUESTION ===
{question}

=== INSTRUCTIONS ===
- Prioritize facts from the retrieved context; cite source numbers (e.g. "According to Source 2...")
- When context is thin or absent, use your expert knowledge of U.S. hazard risk, NRI methodology,
  FEMA programs, and NOAA hazard science to provide a complete, useful answer
- Label any information not from retrieved sources as general domain knowledge
- Be concise but complete; do not refuse to answer just because context is limited"""


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
