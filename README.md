# AI Document Chat Backend

A backend service for uploading documents and chatting with an AI assistant about them, built around a
retrieval pipeline with cross-encoder reranking and source citations. See `tech-spec.md` for the full spec.

## Architecture

```
app/
    api/             FastAPI routes, request/response schemas, dependency wiring
    core/            settings (env-driven) and logging
    domain/          Document/Chunk pydantic models
    interfaces/      abstract contracts (DocumentParser, DocumentChunker, Embedder,
                      VectorStore, Reranker, ChatLLM)
    implementations/ concrete adapters: Docling parser, hybrid chunker, BGE embedder,
                      Qdrant store, BGE cross-encoder reranker, Claude LLM
    services/        orchestration: ingestion pipeline, retrieval pipeline, prompt builder
tests/
    unit/            one test module per component, heavy dependencies mocked
    integration/      upload -> chat flow against a real (temp-path) Qdrant store
```

Everything downstream of `app/interfaces` is swappable — e.g. replacing BGE with another
embedding provider only touches `app/implementations/bge_embedder.py` and the wiring in
`app/api/dependencies.py`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then set ANTHROPIC_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

Docs at `http://localhost:8000/docs`.

## Test

```bash
pytest
```

## Configuration

All tunables are environment variables (see `.env.example`): embedding/rerank model names,
Qdrant storage path, chunk size/overlap, retrieval top-k, final reranked k, and the Claude model.
