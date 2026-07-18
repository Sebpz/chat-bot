from unittest.mock import MagicMock, patch

from app.domain.chunk import Chunk, ScoredChunk
from app.implementations.bge_reranker import BGEReranker


def _make_scored_chunk(chunk_id: str, text: str, retrieval_score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="Handbook.pdf",
        page_number=1,
        heading="Intro",
        text=text,
        token_count=10,
    )
    return ScoredChunk(chunk=chunk, score=retrieval_score)


def test_rerank_sorts_descending_and_truncates_to_top_k() -> None:
    candidates = [
        _make_scored_chunk("c1", "alpha", retrieval_score=0.9),
        _make_scored_chunk("c2", "beta", retrieval_score=0.1),
        _make_scored_chunk("c3", "gamma", retrieval_score=0.5),
    ]

    mock_model = MagicMock()
    # Cross-encoder scores intentionally invert the retrieval-stage ordering.
    mock_model.predict.return_value = [0.2, 0.95, 0.6]

    with patch("app.implementations.bge_reranker.CrossEncoder", return_value=mock_model) as mock_cls:
        reranker = BGEReranker(model_name="fake-model")
        result = reranker.rerank("query", candidates, top_k=2)

    mock_cls.assert_called_once_with("fake-model")
    mock_model.predict.assert_called_once_with(
        [("query", "alpha"), ("query", "beta"), ("query", "gamma")]
    )

    assert [sc.chunk.chunk_id for sc in result] == ["c2", "c3"]
    assert [sc.score for sc in result] == [0.95, 0.6]
    # New cross-encoder score must replace the original retrieval-stage score.
    assert result[0].score != 0.1


def test_rerank_lazy_loads_model_once() -> None:
    candidates = [_make_scored_chunk("c1", "alpha", retrieval_score=0.9)]
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.5]

    with patch("app.implementations.bge_reranker.CrossEncoder", return_value=mock_model) as mock_cls:
        reranker = BGEReranker(model_name="fake-model")
        mock_cls.assert_not_called()
        reranker.rerank("query", candidates, top_k=1)
        reranker.rerank("query", candidates, top_k=1)

    mock_cls.assert_called_once()


def test_rerank_empty_candidates_short_circuits() -> None:
    with patch("app.implementations.bge_reranker.CrossEncoder") as mock_cls:
        reranker = BGEReranker(model_name="fake-model")
        result = reranker.rerank("query", [], top_k=5)

    assert result == []
    mock_cls.assert_not_called()
