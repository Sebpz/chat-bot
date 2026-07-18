from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    chunks: int


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int


class ChatRequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    document: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
