# RAG System Design

## 1. Document Corpus
Stored in:
s3://hazard/docs/

Sources:
- FEMA Disaster Reports (PDF)
- NOAA event narratives
- NRI documentation
- State emergency plans (optional)

---

## 2. Chunking
Script: rag/indexing/chunk_documents.py

Chunk size: 512 tokens  
Overlap: 64 tokens

Metadata stored per chunk:
- doc_id
- section
- hazard_type
- county references (if extracted)

---

## 3. Embeddings
Model: 
- Amazon Titan Embeddings (recommended)
OR
- HuggingFace Instructor-XL

Embedding script:
rag/indexing/embed_and_index.py

---

## 4. Vector Store
OpenSearch Serverless collection:
- Vector index with 1536-dim embedding
- kNN search enabled
- Schema:
  - id
  - embedding
  - text
  - metadata

---

## 5. Retrieval
rag/retrieval/retrieve.py

Query → embedding → top-k (k=5–10) → context

---

## 6. LLM Generation
Options:
- Bedrock Claude 3 Sonnet (preferred)
- Llama3.1 on ECS for local/open-source

Prompt structure:
- System: domain persona
- Context: retrieved chunks
- User question

---

## 7. Unified API Workflow

User → `/ask` →
1. Embed question
2. Retrieve top chunks
3. Construct prompt
4. Query LLM
5. Return:
   - answer
   - referenced docs
   - model risk prediction (optional)
