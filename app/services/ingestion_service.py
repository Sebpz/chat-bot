import uuid

from app.core.logging import get_logger, log_duration
from app.interfaces.chunker import DocumentChunker
from app.interfaces.embedder import Embedder
from app.interfaces.parser import DocumentParser
from app.interfaces.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionService:
    """Orchestrates the upload pipeline: parse -> chunk -> embed -> store."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: DocumentChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest(self, file_path: str, filename: str) -> tuple[str, int]:
        document_id = str(uuid.uuid4())

        with log_duration(logger, "parsing", document_id=document_id):
            document = self._parser.parse(file_path, document_id, filename)

        with log_duration(logger, "chunking", document_id=document_id):
            chunks = self._chunker.chunk(document)

        if not chunks:
            self._vector_store.register_document(document_id, filename, 0)
            return document_id, 0

        with log_duration(logger, "embedding", document_id=document_id, chunk_count=len(chunks)):
            embeddings = self._embedder.embed_texts([chunk.text for chunk in chunks])

        with log_duration(logger, "vector_store_upsert", document_id=document_id):
            self._vector_store.upsert_chunks(chunks, embeddings)
            self._vector_store.register_document(document_id, filename, len(chunks))

        return document_id, len(chunks)
