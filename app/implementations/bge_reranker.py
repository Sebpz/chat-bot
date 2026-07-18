from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.domain.chunk import ScoredChunk
from app.interfaces.reranker import Reranker
from app.utils.errors import RerankingError


class BGEReranker(Reranker):
    """Cross-encoder reranker backed by BAAI/bge-reranker-base (or a configured alternative)."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().rerank_model
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        try:
            model = self._get_model()
            raw_scores = model.predict(pairs)
        except Exception as exc:
            raise RerankingError(f"Cross-encoder reranking failed: {exc}") from exc

        rescored = [
            ScoredChunk(chunk=candidate.chunk, score=float(score))
            for candidate, score in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda sc: sc.score, reverse=True)
        return rescored[:top_k]
