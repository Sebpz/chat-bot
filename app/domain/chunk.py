from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None = None
    heading: str | None = None
    text: str
    token_count: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
