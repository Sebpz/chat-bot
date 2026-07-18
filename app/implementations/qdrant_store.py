import uuid
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import get_settings
from app.domain.chunk import Chunk, ScoredChunk
from app.interfaces.vector_store import VectorStore
from app.utils.errors import VectorStoreError

CHUNKS_COLLECTION = "chunks"
DOCUMENTS_COLLECTION = "documents"
_DOCUMENTS_VECTOR_SIZE = 1  # documents collection stores metadata only, not embeddings


def _point_id(key: str) -> str:
    """Qdrant point IDs must be an unsigned int or UUID; derive a stable UUID from our string keys."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantVectorStore(VectorStore):
    def __init__(self, path: str | None = None, vector_size: int = 384):
        self._path = path or get_settings().qdrant_path
        self._vector_size = vector_size
        try:
            self._client = QdrantClient(path=self._path)
        except Exception as exc:
            raise VectorStoreError(f"Failed to open Qdrant store at '{self._path}': {exc}") from exc
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        try:
            if not self._client.collection_exists(CHUNKS_COLLECTION):
                self._client.create_collection(
                    collection_name=CHUNKS_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=self._vector_size, distance=qmodels.Distance.COSINE
                    ),
                )
            if not self._client.collection_exists(DOCUMENTS_COLLECTION):
                self._client.create_collection(
                    collection_name=DOCUMENTS_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=_DOCUMENTS_VECTOR_SIZE, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize Qdrant collections: {exc}") from exc

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("chunks and embeddings must be the same length")
        if not chunks:
            return
        try:
            points = [
                qmodels.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "page_number": chunk.page_number,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "token_count": chunk.token_count,
                        "created_at": chunk.created_at.isoformat(),
                        "metadata": chunk.metadata,
                    },
                )
                for chunk, embedding in zip(chunks, embeddings)
            ]
            self._client.upsert(collection_name=CHUNKS_COLLECTION, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert {len(chunks)} chunk(s): {exc}") from exc

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        query_filter = None
        if document_ids:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_id", match=qmodels.MatchAny(any=document_ids)
                    )
                ]
            )
        try:
            results = self._client.query_points(
                collection_name=CHUNKS_COLLECTION,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

        return [self._to_scored_chunk(point) for point in results]

    @staticmethod
    def _to_scored_chunk(point) -> ScoredChunk:
        payload = point.payload
        chunk = Chunk(
            chunk_id=payload["chunk_id"],
            document_id=payload["document_id"],
            document_name=payload["document_name"],
            page_number=payload.get("page_number"),
            heading=payload.get("heading"),
            text=payload["text"],
            token_count=payload["token_count"],
            created_at=payload["created_at"],
            metadata=payload.get("metadata") or {},
        )
        return ScoredChunk(chunk=chunk, score=point.score)

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=CHUNKS_COLLECTION,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_id", match=qmodels.MatchValue(value=document_id)
                            )
                        ]
                    )
                ),
            )
            self._client.delete(
                collection_name=DOCUMENTS_COLLECTION,
                points_selector=qmodels.PointIdsList(points=[_point_id(document_id)]),
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{document_id}': {exc}") from exc

    def list_documents(self) -> list[dict]:
        try:
            records, _ = self._client.scroll(
                collection_name=DOCUMENTS_COLLECTION, limit=10_000, with_payload=True
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to list documents: {exc}") from exc
        return [record.payload for record in records]

    def register_document(self, document_id: str, filename: str, chunk_count: int) -> None:
        try:
            self._client.upsert(
                collection_name=DOCUMENTS_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=_point_id(document_id),
                        vector=[0.0] * _DOCUMENTS_VECTOR_SIZE,
                        payload={
                            "document_id": document_id,
                            "filename": filename,
                            "chunk_count": chunk_count,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                ],
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to register document '{document_id}': {exc}") from exc
