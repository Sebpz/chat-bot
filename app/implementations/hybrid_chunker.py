from __future__ import annotations

from uuid import uuid4

from app.core.config import get_settings
from app.domain.chunk import Chunk
from app.domain.document import Document, Figure, Paragraph, Section, Table
from app.interfaces.chunker import DocumentChunker
from app.utils.tokens import count_tokens, split_by_tokens


def _element_text(element: Paragraph | Table | Figure) -> str:
    if isinstance(element, Figure):
        return element.caption or ""
    return element.text


def _first_page_number(section: Section) -> int | None:
    for element in section.elements:
        if element.page_number is not None:
            return element.page_number
    return None


class HybridDocumentChunker(DocumentChunker):
    """Structure-aware chunker: splits within section/heading boundaries first,
    falling back to a flat token-window split when a document has no sections.

    Docling's own HybridChunker operates on docling's native document object,
    which DocumentParser implementations already convert away from before this
    chunker ever sees the data — so this reimplements the same spirit (respect
    structure, fall back to token windows) directly against our own Document model.
    """

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        if document.sections:
            chunks = self._chunk_sections(document)
            if chunks:
                return chunks

        return self._chunk_flat(document)

    def _chunk_sections(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            text = "\n".join(_element_text(element) for element in section.elements).strip()
            if not text:
                continue

            page_number = _first_page_number(section)
            windows = split_by_tokens(text, self._chunk_size, self._chunk_overlap)
            for window in windows:
                if not window.strip():
                    continue
                chunks.append(self._build_chunk(document, window, heading=section.heading, page_number=page_number))
        return chunks

    def _chunk_flat(self, document: Document) -> list[Chunk]:
        all_text = "\n".join(
            _element_text(element) for section in document.sections for element in section.elements
        ).strip()
        if not all_text:
            return []

        windows = split_by_tokens(all_text, self._chunk_size, self._chunk_overlap)
        return [
            self._build_chunk(document, window, heading=None, page_number=None)
            for window in windows
            if window.strip()
        ]

    def _build_chunk(
        self, document: Document, text: str, *, heading: str | None, page_number: int | None
    ) -> Chunk:
        return Chunk(
            chunk_id=str(uuid4()),
            document_id=document.document_id,
            document_name=document.filename,
            page_number=page_number,
            heading=heading,
            text=text,
            token_count=count_tokens(text),
            metadata={},
        )
