from unittest.mock import MagicMock, patch

import pytest

from app.implementations.bge_embedder import _QUERY_INSTRUCTION, BGEEmbedder
from app.utils.errors import EmbeddingError


class _FakeArray:
    """Stand-in for the numpy array sentence-transformers returns from encode()."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


def test_constructor_does_not_load_model():
    with patch("app.implementations.bge_embedder.SentenceTransformer") as mock_st:
        BGEEmbedder(model_name="some/model")
        mock_st.assert_not_called()


def test_embed_texts_calls_encode_with_no_prefix_and_normalization():
    with patch("app.implementations.bge_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = _FakeArray([[0.1, 0.2], [0.3, 0.4]])
        mock_st.return_value = mock_model

        embedder = BGEEmbedder(model_name="some/model")
        result = embedder.embed_texts(["hello world", "second doc"])

        mock_model.encode.assert_called_once_with(
            ["hello world", "second doc"], normalize_embeddings=True
        )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_st.assert_called_once_with("some/model")


def test_embed_query_prepends_instruction_prefix():
    with patch("app.implementations.bge_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = _FakeArray([0.5, 0.6])
        mock_st.return_value = mock_model

        embedder = BGEEmbedder(model_name="some/model")
        result = embedder.embed_query("what is the leave policy?")

        mock_model.encode.assert_called_once_with(
            _QUERY_INSTRUCTION + "what is the leave policy?", normalize_embeddings=True
        )
        assert result == [0.5, 0.6]


def test_dimension_returns_and_caches_model_dimension():
    with patch("app.implementations.bge_embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        embedder = BGEEmbedder(model_name="some/model")
        assert embedder.dimension == 384
        assert embedder.dimension == 384
        mock_model.get_sentence_embedding_dimension.assert_called_once()


def test_default_model_name_comes_from_settings():
    with patch("app.implementations.bge_embedder.get_settings") as mock_settings:
        mock_settings.return_value.embedding_model = "BAAI/bge-small-en-v1.5"
        embedder = BGEEmbedder()
        assert embedder._model_name == "BAAI/bge-small-en-v1.5"


def test_embed_texts_wraps_failures_in_embedding_error():
    with patch("app.implementations.bge_embedder.SentenceTransformer") as mock_st:
        mock_st.side_effect = RuntimeError("boom")
        embedder = BGEEmbedder(model_name="some/model")
        with pytest.raises(EmbeddingError):
            embedder.embed_texts(["x"])
