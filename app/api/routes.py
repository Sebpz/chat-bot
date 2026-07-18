import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.dependencies import (
    get_ingestion_service,
    get_llm,
    get_retrieval_service,
    get_vector_store,
)
from app.api.schemas import ChatRequest, ChatResponse, Citation, DocumentSummary, UploadResponse
from app.core.logging import get_logger, log_duration
from app.implementations.claude_llm import ClaudeChatLLM
from app.implementations.qdrant_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.prompt_builder import SYSTEM_PROMPT, build_context, build_user_prompt
from app.services.retrieval_service import RetrievalService
from app.utils.errors import DocumentNotFoundError

logger = get_logger(__name__)
router = APIRouter()


@router.post("/documents", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with log_duration(logger, "upload_total", filename=file.filename):
            document_id, chunk_count = ingestion_service.ingest(tmp_path, file.filename or "unknown")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return UploadResponse(document_id=document_id, chunks=chunk_count)


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> list[DocumentSummary]:
    return [DocumentSummary(**doc) for doc in vector_store.list_documents()]


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    vector_store: QdrantVectorStore = Depends(get_vector_store),
) -> None:
    existing_ids = {doc["document_id"] for doc in vector_store.list_documents()}
    if document_id not in existing_ids:
        raise DocumentNotFoundError(f"Document {document_id} not found")
    vector_store.delete_document(document_id)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm: ClaudeChatLLM = Depends(get_llm),
) -> ChatResponse:
    with log_duration(logger, "chat_total", question=request.question):
        reranked = retrieval_service.retrieve(request.question, request.document_ids or None)

        if not reranked:
            return ChatResponse(
                answer="I could not find relevant information in the supplied documents.",
                citations=[],
            )

        context = build_context(reranked)
        user_prompt = build_user_prompt(request.question, context)

        with log_duration(logger, "llm_generation"):
            answer = llm.generate(SYSTEM_PROMPT, user_prompt)

        seen: set[tuple[str, int | None]] = set()
        citations: list[Citation] = []
        for scored in reranked:
            key = (scored.chunk.document_name, scored.chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(document=scored.chunk.document_name, page=scored.chunk.page_number))

    return ChatResponse(answer=answer, citations=citations)
