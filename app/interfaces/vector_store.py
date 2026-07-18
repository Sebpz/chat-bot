from abc import ABC, abstractmethod

from app.domain.chunk import Chunk, ScoredChunk


class VectorStore(ABC):
    """Persists chunk embeddings and serves dense similarity search."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self) -> list[dict]:
        """Return summaries like {"document_id": ..., "filename": ..., "chunk_count": ...}."""
        raise NotImplementedError

    @abstractmethod
    def register_document(self, document_id: str, filename: str, chunk_count: int) -> None:
        raise NotImplementedError
