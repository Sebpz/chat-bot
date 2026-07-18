from app.core.config import get_settings
from app.core.logging import get_logger, log_duration
from app.domain.chunk import ScoredChunk
from app.interfaces.embedder import Embedder
from app.interfaces.reranker import Reranker
from app.interfaces.vector_store import VectorStore

logger = get_logger(__name__)


class RetrievalService:
    """Orchestrates the chat pipeline's retrieval half: embed -> search -> rerank."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore, reranker: Reranker) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> list[ScoredChunk]:
        settings = get_settings()

        with log_duration(logger, "query_embedding"):
            query_embedding = self._embedder.embed_query(question)

        with log_duration(logger, "retrieval", top_k=settings.top_k):
            candidates = self._vector_store.search(query_embedding, settings.top_k, document_ids)

        if not candidates:
            return []

        with log_duration(logger, "reranking", final_k=settings.final_k):
            reranked = self._reranker.rerank(question, candidates, settings.final_k)

        return reranked
