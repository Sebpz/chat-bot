import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.api import dependencies as deps
from app.domain.document import Document, Paragraph, Section
from app.implementations.hybrid_chunker import HybridDocumentChunker
from app.implementations.qdrant_store import QdrantVectorStore
from app.interfaces.embedder import Embedder
from app.interfaces.llm import ChatLLM
from app.interfaces.parser import DocumentParser
from app.interfaces.reranker import Reranker
from app.main import app
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

VECTOR_SIZE = 4


class FakeParser(DocumentParser):
    """Skips real Docling parsing: wraps the uploaded file's raw text in one section."""

    def parse(self, file_path: str, document_id: str, filename: str) -> Document:
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return Document(
            document_id=document_id,
            filename=filename,
            sections=[Section(heading="Annual Leave", elements=[Paragraph(text=text, page_number=14)])],
        )


class FakeEmbedder(Embedder):
    """Deterministic low-dimension embeddings so real Qdrant search stays exact and fast."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def dimension(self) -> int:
        return VECTOR_SIZE

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0] if "leave" in text.lower() else [0.0, 1.0, 0.0, 0.0]


class PassthroughReranker(Reranker):
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class FakeLLM(ChatLLM):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return "Employees receive 20 days of annual leave, according to the handbook."


@pytest.fixture
def client():
    tmp_path = tempfile.mkdtemp(prefix="qdrant_integration_")
    vector_store = QdrantVectorStore(path=tmp_path, vector_size=VECTOR_SIZE)
    ingestion_service = IngestionService(
        parser=FakeParser(),
        chunker=HybridDocumentChunker(chunk_size=900, chunk_overlap=150),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )
    retrieval_service = RetrievalService(
        embedder=FakeEmbedder(), vector_store=vector_store, reranker=PassthroughReranker()
    )

    app.dependency_overrides[deps.get_ingestion_service] = lambda: ingestion_service
    app.dependency_overrides[deps.get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[deps.get_vector_store] = lambda: vector_store
    app.dependency_overrides[deps.get_llm] = lambda: FakeLLM()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_upload_then_chat_returns_answer_with_citations(client):
    upload_response = client.post(
        "/documents",
        files={"file": ("handbook.md", b"Employees receive 20 days of annual leave per year.", "text/markdown")},
    )
    assert upload_response.status_code == 200
    body = upload_response.json()
    assert body["chunks"] > 0
    document_id = body["document_id"]

    chat_response = client.post("/chat", json={"question": "How many days of annual leave do I get?"})
    assert chat_response.status_code == 200
    chat_body = chat_response.json()

    assert chat_body["answer"]
    assert len(chat_body["citations"]) > 0
    assert chat_body["citations"][0]["document"] == "handbook.md"
    assert chat_body["citations"][0]["page"] == 14

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert any(doc["document_id"] == document_id for doc in list_response.json())

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    list_after_delete = client.get("/documents")
    assert all(doc["document_id"] != document_id for doc in list_after_delete.json())


def test_chat_with_no_matching_documents_returns_no_citations(client):
    response = client.post("/chat", json={"question": "anything at all"})
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert "could not find" in body["answer"].lower()


def test_delete_unknown_document_returns_404(client):
    response = client.delete("/documents/does-not-exist")
    assert response.status_code == 404
