from abc import ABC, abstractmethod

from app.domain.chunk import Chunk
from app.domain.document import Document


class DocumentChunker(ABC):
    """Splits a parsed Document into retrieval-sized Chunks."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
