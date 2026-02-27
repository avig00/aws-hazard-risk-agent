"""
Document chunking pipeline for the RAG corpus.

Reads FEMA disaster reports, NOAA narratives, and NRI documentation
from S3 (or local paths), extracts text, and splits into overlapping
chunks with metadata for indexing into OpenSearch.

Supported formats: PDF, TXT, Markdown
"""
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Iterator

import boto3
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "rag_config.yml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _doc_id(source_path: str, chunk_index: int) -> str:
    """Deterministic ID so re-indexing is idempotent."""
    key = f"{source_path}::{chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def extract_text_from_pdf(content: bytes) -> str:
    """Extract plain text from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        raise ImportError("pypdf required: pip install pypdf")


def extract_text(content: bytes, extension: str) -> str:
    """Route to the correct extractor by file extension."""
    if extension == ".pdf":
        return extract_text_from_pdf(content)
    else:
        # .txt or .md
        return content.decode("utf-8", errors="replace")


def token_aware_chunks(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping character-level chunks that approximate
    token boundaries (rough heuristic: 1 token ≈ 4 chars).
    Returns list of chunk strings.
    """
    char_chunk = chunk_size * 4
    char_overlap = overlap * 4

    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()

    chunks = []
    start = 0
    while start < len(text):
        end = start + char_chunk
        chunks.append(text[start:end])
        start += char_chunk - char_overlap
        if start >= len(text):
            break
    return chunks


def infer_hazard_type(text: str) -> str:
    """Rough keyword extraction for hazard_type metadata."""
    text_lower = text.lower()
    for hazard in ["flood", "hurricane", "tornado", "wildfire", "earthquake", "drought",
                   "hail", "winter storm", "thunderstorm"]:
        if hazard in text_lower:
            return hazard
    return "general"


def process_document(
    source_path: str,
    content: bytes,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list:
    """
    Extract and chunk one document.

    Returns list of dicts:
    {
        id, text, metadata: {source, section, hazard_type, chunk_index, total_chunks}
    }
    """
    ext = Path(source_path).suffix.lower()
    raw_text = extract_text(content, ext)

    if not raw_text.strip():
        logger.warning("Empty text from %s — skipping", source_path)
        return []

    raw_chunks = token_aware_chunks(raw_text, chunk_size, overlap)
    total = len(raw_chunks)

    records = []
    for i, chunk_text in enumerate(raw_chunks):
        records.append({
            "id": _doc_id(source_path, i),
            "text": chunk_text,
            "metadata": {
                "source": source_path,
                "section": f"chunk_{i}",
                "hazard_type": infer_hazard_type(chunk_text),
                "chunk_index": i,
                "total_chunks": total,
            },
        })

    logger.info("Chunked %s → %d chunks", source_path, total)
    return records


def load_corpus_from_s3(bucket: str, prefix: str, extensions: list) -> Iterator[tuple]:
    """
    Yield (s3_key, content_bytes) for each document in the S3 corpus prefix.
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if any(key.lower().endswith(ext) for ext in extensions):
                response = s3.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read()
                yield f"s3://{bucket}/{key}", content


def load_corpus_from_local(directory: str, extensions: list) -> Iterator[tuple]:
    """Yield (path_str, content_bytes) for local files."""
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in extensions:
            yield str(path), path.read_bytes()


def build_corpus_chunks(config: dict = None) -> list:
    """
    Load the full corpus from S3, chunk all documents, return flat list of records.
    """
    if config is None:
        config = load_config()

    rag_cfg = config["rag"]
    corpus_cfg = config.get("corpus", {})

    chunk_size = rag_cfg["chunk_size"]
    overlap = rag_cfg["chunk_overlap"]
    extensions = corpus_cfg.get("supported_extensions", [".pdf", ".txt", ".md"])

    bucket = corpus_cfg.get("s3_bucket", "hazard")
    prefix = corpus_cfg.get("s3_prefix", "docs/")

    all_chunks = []
    for source_path, content in load_corpus_from_s3(bucket, prefix, extensions):
        chunks = process_document(source_path, content, chunk_size, overlap)
        all_chunks.extend(chunks)

    logger.info("Total chunks built: %d from corpus %s/%s", len(all_chunks), bucket, prefix)
    return all_chunks


if __name__ == "__main__":
    chunks = build_corpus_chunks()
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print("Sample chunk:", chunks[0]["text"][:200])
        print("Metadata:", chunks[0]["metadata"])
