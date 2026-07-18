from functools import lru_cache

from app.core.config import get_settings
from app.implementations.bge_embedder import BGEEmbedder
from app.implementations.bge_reranker import BGEReranker
from app.implementations.claude_llm import ClaudeChatLLM
from app.implementations.docling_parser import DoclingParser
from app.implementations.hybrid_chunker import HybridDocumentChunker
from app.implementations.qdrant_store import QdrantVectorStore
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

# Every implementation below is expensive to construct (models, DB handles), so
# each is wired as a process-wide singleton via lru_cache rather than per-request.


@lru_cache
def get_parser() -> DoclingParser:
    return DoclingParser()


@lru_cache
def get_chunker() -> HybridDocumentChunker:
    return HybridDocumentChunker()


@lru_cache
def get_embedder() -> BGEEmbedder:
    return BGEEmbedder()


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(path=settings.qdrant_path, vector_size=get_embedder().dimension)


@lru_cache
def get_reranker() -> BGEReranker:
    return BGEReranker()


@lru_cache
def get_llm() -> ClaudeChatLLM:
    return ClaudeChatLLM()


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(get_parser(), get_chunker(), get_embedder(), get_vector_store())


@lru_cache
def get_retrieval_service() -> RetrievalService:
    return RetrievalService(get_embedder(), get_vector_store(), get_reranker())
