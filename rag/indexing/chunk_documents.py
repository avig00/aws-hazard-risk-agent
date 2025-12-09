import os

def chunk_text(text: str, chunk_size=512, overlap=64):
    """
    Create overlapping chunks from long text.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def process_document(path: str):
    """
    Read PDF or text, extract raw text, chunk into segments.
    """
    print(f"Chunking document: {path}")
    # Placeholder extraction
    text = open(path).read()
    return chunk_text(text)

if __name__ == "__main__":
    pass
