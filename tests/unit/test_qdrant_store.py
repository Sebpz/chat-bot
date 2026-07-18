import shutil
import tempfile

import pytest

from app.domain.chunk import Chunk
from app.implementations.qdrant_store import QdrantVectorStore

VECTOR_SIZE = 4


@pytest.fixture
def store():
    tmp_path = tempfile.mkdtemp(prefix="qdrant_test_")
    store = QdrantVectorStore(path=tmp_path, vector_size=VECTOR_SIZE)
    yield store
    shutil.rmtree(tmp_path, ignore_errors=True)


def _chunk(chunk_id: str, document_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        page_number=1,
        heading="Intro",
        text=text,
        token_count=3,
    )


def test_upsert_and_search_returns_scored_chunks(store):
    chunks = [_chunk("c1", "doc1"), _chunk("c2", "doc1")]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    store.upsert_chunks(chunks, embeddings)

    results = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].score >= results[1].score
    assert results[0].chunk.document_name == "doc1.pdf"


def test_search_filters_by_document_ids(store):
    store.upsert_chunks([_chunk("c1", "doc1")], [[1.0, 0.0, 0.0, 0.0]])
    store.upsert_chunks([_chunk("c2", "doc2")], [[1.0, 0.0, 0.0, 0.0]])

    results = store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5, document_ids=["doc2"])

    assert len(results) == 1
    assert results[0].chunk.document_id == "doc2"


def test_register_and_list_documents(store):
    store.register_document("doc1", "handbook.pdf", chunk_count=10)
    store.register_document("doc2", "policy.pdf", chunk_count=5)

    documents = store.list_documents()
    document_ids = {d["document_id"] for d in documents}

    assert document_ids == {"doc1", "doc2"}


def test_delete_document_removes_chunks_and_metadata(store):
    store.register_document("doc1", "handbook.pdf", chunk_count=1)
    store.upsert_chunks([_chunk("c1", "doc1")], [[1.0, 0.0, 0.0, 0.0]])

    store.delete_document("doc1")

    assert store.search(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=5) == []
    assert store.list_documents() == []


def test_upsert_mismatched_lengths_raises():
    from app.utils.errors import VectorStoreError

    tmp_path = tempfile.mkdtemp(prefix="qdrant_test_")
    try:
        store = QdrantVectorStore(path=tmp_path, vector_size=VECTOR_SIZE)
        with pytest.raises(VectorStoreError):
            store.upsert_chunks([_chunk("c1", "doc1")], [])
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
