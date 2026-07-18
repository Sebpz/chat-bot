from app.domain.chunk import Chunk, ScoredChunk
from app.services.retrieval_service import RetrievalService


def _scored(chunk_id: str, score: float) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="handbook.pdf",
        page_number=1,
        heading="Intro",
        text=f"text for {chunk_id}",
        token_count=3,
    )
    return ScoredChunk(chunk=chunk, score=score)


class FakeEmbedder:
    def __init__(self):
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts):  # pragma: no cover - unused by retrieval
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        return 3


class FakeVectorStore:
    def __init__(self, results: list[ScoredChunk]):
        self._results = results
        self.calls: list[dict] = []

    def search(self, query_embedding, top_k, document_ids=None):
        self.calls.append({"query_embedding": query_embedding, "top_k": top_k, "document_ids": document_ids})
        return self._results

    def upsert_chunks(self, chunks, embeddings):  # pragma: no cover - unused
        raise NotImplementedError

    def delete_document(self, document_id):  # pragma: no cover - unused
        raise NotImplementedError

    def list_documents(self):  # pragma: no cover - unused
        raise NotImplementedError

    def register_document(self, document_id, filename, chunk_count):  # pragma: no cover - unused
        raise NotImplementedError


class FakeReranker:
    def __init__(self):
        self.rerank_calls: list[tuple] = []

    def rerank(self, query, candidates, top_k):
        self.rerank_calls.append((query, candidates, top_k))
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_k]


def test_retrieve_embeds_searches_and_reranks_in_order():
    candidates = [_scored("c1", 0.5), _scored("c2", 0.9), _scored("c3", 0.1)]
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore(candidates)
    reranker = FakeReranker()

    service = RetrievalService(embedder, vector_store, reranker)
    results = service.retrieve("What is the leave policy?")

    assert embedder.queries == ["What is the leave policy?"]
    assert vector_store.calls[0]["query_embedding"] == [0.1, 0.2, 0.3]
    assert reranker.rerank_calls[0][0] == "What is the leave policy?"
    assert [c.chunk.chunk_id for c in results] == ["c2", "c1", "c3"]


def test_retrieve_passes_document_ids_filter_through_to_search():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore([_scored("c1", 1.0)])
    reranker = FakeReranker()

    service = RetrievalService(embedder, vector_store, reranker)
    service.retrieve("question", document_ids=["doc-1", "doc-2"])

    assert vector_store.calls[0]["document_ids"] == ["doc-1", "doc-2"]


def test_retrieve_returns_empty_without_reranking_when_no_candidates():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore([])
    reranker = FakeReranker()

    service = RetrievalService(embedder, vector_store, reranker)
    results = service.retrieve("question")

    assert results == []
    assert reranker.rerank_calls == []
