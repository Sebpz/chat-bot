Based on what you've told me about the previous interview, I agree. If they spent time discussing **RAG pipelines, hybrid retrieval, cross-encoder reranking, observability, and production AI systems**, then including reranking is likely to score points. It shows you're building something closer to a production retrieval stack rather than a tutorial.

I'd also bias the implementation toward **showing good software engineering**, not just getting the chatbot working.

---

# Technical Specification

## AI Document Chat Backend

### Objective

Build a backend service that allows users to upload documents and chat with an AI assistant about those documents.

The system should emphasize:

* modular architecture
* high retrieval quality
* production-ready design
* clean APIs
* source attribution
* maintainability

---

# High-Level Architecture

```text
                        Upload Document
                               │
                               ▼
                     Document Processing Service
                               │
          ┌────────────────────┴───────────────────┐
          │                                        │
          ▼                                        ▼
      Docling Parser                      Metadata Extractor
          │
          ▼
   Hybrid Document Chunker
          │
          ▼
    Embedding Generator
          │
          ▼
      Vector Database
          │
          ▼
        Retriever
(Dense Similarity Search Top 20)
          │
          ▼
 Cross Encoder Reranker
        (Top 20 → Top 5)
          │
          ▼
 Prompt Construction
          │
          ▼
      Claude Chat API
          │
          ▼
 JSON Response + Citations
```

---

# Technology Stack

## Language

Python 3.11+

---

## API

FastAPI

Reasons

* async support
* OpenAPI
* Pydantic
* production ready

---

## Document Parsing

Docling

Responsibilities

* parse PDF
* DOCX
* Markdown
* preserve headings
* preserve tables
* preserve page numbers

Output

```python
Document

Section

Paragraph

Table

Figure
```

---

## Chunking

### Preferred

Docling HybridChunker

Fallback

Recursive token splitter

Configuration

```text
Chunk Size

900 tokens

Overlap

150 tokens
```

Each chunk should contain

```python
class Chunk:

    chunk_id

    document_id

    page_number

    heading

    text

    metadata
```

---

# Embedding

Preferred

BAAI/bge-small-en-v1.5

Reasons

* CPU friendly

* excellent benchmark performance

* widely adopted

Alternative

bge-base

---

# Vector Database

Qdrant Local

```python
QdrantClient(path="./storage")
```

Reasons

* production quality

* persistent

* no Docker

* simple API

Collections

```text
documents

chunks
```

---

# Retrieval

Top K

20

Search

Dense similarity search

Pipeline

```text
Query

↓

Embedding

↓

Vector Search

↓

Top 20
```

---

# Reranking

Mandatory

Cross Encoder

Model

BAAI/bge-reranker-base

Pipeline

```text
Top 20

↓

Cross Encoder

↓

Top 5
```

Output

```python
Chunk

Score
```

Sorted descending.

---

# Context Construction

Merge reranked chunks

Maximum context

6000 tokens

Context format

```text
Document

Page

Heading

Content
```

Example

```
Document: Employee Handbook

Page: 14

Section: Annual Leave

Employees receive...
```

---

# Prompt Template

System

```
You answer questions using ONLY the supplied context.

If the answer cannot be found, clearly state that.

Always cite the document and page number.
```

User

```
Question

Context
```

---

# Chat Model

Claude

Temperature

0

Streaming optional.

---

# API Endpoints

## Upload

```
POST

/documents
```

Input

Multipart

Output

```json
{
  "document_id": "...",
  "chunks": 143
}
```

---

## Chat

```
POST

/chat
```

Input

```json
{
  "question": "...",
  "document_ids":[]
}
```

Output

```json
{
    "answer": "...",
    "citations":[
        {
            "document":"...",
            "page":4
        }
    ]
}
```

---

## List Documents

```
GET

/documents
```

---

## Delete Document

```
DELETE

/documents/{id}
```

---

# Metadata

Every chunk stores

```python
chunk_id

document_id

document_name

page_number

heading

embedding

token_count

created_at
```

Metadata allows

* filtering

* citations

* debugging

---

# Project Structure

```
app/

    api/

        routes.py

    core/

        config.py

        logging.py

    models/

        document.py

        chunk.py

    services/

        parser.py

        chunker.py

        embeddings.py

        vector_store.py

        retrieval.py

        reranker.py

        prompt_builder.py

        llm.py

    utils/

    main.py
```

---

# Processing Flow

## Upload

```
Receive file

↓

Docling

↓

Hybrid Chunking

↓

Embeddings

↓

Store vectors

↓

Return document id
```

---

## Chat

```
Receive question

↓

Generate embedding

↓

Top 20 retrieval

↓

Cross Encoder reranking

↓

Top 5

↓

Prompt construction

↓

Claude

↓

Response
```

---

# Error Handling

Gracefully handle

Unsupported file

Corrupted PDF

Embedding failure

Vector DB unavailable

LLM timeout

Return

```
HTTPException
```

with informative messages.

---

# Logging

Record

Upload duration

Parsing duration

Chunk count

Embedding duration

Retrieval latency

Rerank latency

LLM latency

Total request latency

Useful for debugging and optimization.

---

# Configuration

Environment variables

```
EMBEDDING_MODEL

RERANK_MODEL

QDRANT_PATH

CHUNK_SIZE

CHUNK_OVERLAP

TOP_K

FINAL_K

CLAUDE_MODEL
```

No hardcoded values.

---

# Testing

Unit tests

Parser

Chunker

Retriever

Reranker

Prompt builder

Integration test

Upload document

Ask question

Verify

* answer generated
* citations returned

---

# Future Extensions (Out of Scope)

* Hybrid BM25 + vector retrieval (Qdrant supports sparse vectors if you want to extend the system later).
* Parent-child retrieval for large documents.
* OCR fallback for scanned PDFs.
* Async embedding queue for bulk uploads.
* Multi-user authentication and document ownership.
* Observability with Langfuse/OpenTelemetry.
* Contextual embeddings (prepend section headings before embedding).

---

# Design Decisions & Trade-offs

| Decision     | Choice                             | Rationale                                                                           |
| ------------ | ---------------------------------- | ----------------------------------------------------------------------------------- |
| Parser       | Docling                            | Preserves document structure and metadata, enabling better chunking and citations.  |
| Chunking     | Docling HybridChunker              | Leverages document hierarchy instead of arbitrary token windows.                    |
| Embeddings   | BGE Small                          | Strong quality-to-speed ratio for CPU environments; easy to swap via configuration. |
| Vector Store | Qdrant Local                       | Production-grade API without external infrastructure.                               |
| Retrieval    | Dense similarity (Top 20)          | Simple, effective baseline that can be extended to hybrid search.                   |
| Reranking    | BGE Cross-Encoder (Top 20 → Top 5) | Significantly improves retrieval precision before LLM generation.                   |
| API          | FastAPI                            | Async, type-safe, and interview-friendly.                                           |
| Architecture | Layered services                   | Keeps parsing, retrieval, reranking, and LLM integration independently testable.    |

## One improvement I'd make

I would **slightly upgrade** this architecture to demonstrate stronger engineering maturity without adding much complexity:

```
app/
├── api/
├── core/
├── domain/          # Business models (Document, Chunk)
├── services/
│   ├── ingestion/
│   ├── retrieval/
│   ├── llm/
│   └── storage/
├── interfaces/      # Abstract base classes
├── implementations/ # Concrete implementations (Docling, Qdrant, BGE)
└── tests/
```

By introducing interfaces like `DocumentParser`, `Embedder`, `VectorStore`, and `Reranker`, you make components easily swappable (e.g., replacing BGE with Voyage AI or Qdrant with pgvector). This demonstrates familiarity with dependency inversion and clean architecture principles—something that tends to stand out positively in senior AI engineering interviews without significantly increasing implementation effort.
