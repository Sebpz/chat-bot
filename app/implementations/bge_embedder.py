from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.interfaces.embedder import Embedder
from app.utils.errors import EmbeddingError

# BGE models are trained so that, for asymmetric retrieval, only the query side
# gets this instruction prefix — document/passage embeddings are left as-is.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder(Embedder):
    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or get_settings().embedding_model
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(self._model_name)
            except Exception as exc:
                raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {exc}") from exc
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            model = self._get_model()
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed {len(texts)} text(s): {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            model = self._get_model()
            embedding = model.encode(_QUERY_INSTRUCTION + text, normalize_embeddings=True)
            return embedding.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._get_model().get_sentence_embedding_dimension()
        return self._dimension
