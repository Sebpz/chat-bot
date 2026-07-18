from abc import ABC, abstractmethod

from app.domain.chunk import ScoredChunk


class Reranker(ABC):
    """Cross-encoder reranking of candidate chunks against the query."""

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the top_k candidates re-scored and sorted descending by relevance."""
        raise NotImplementedError
