from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel

from app.domain.document import Document, Figure, Paragraph, Section, Table
from app.interfaces.parser import DocumentParser
from app.utils.errors import ParsingError, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md"}

_HEADER_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
_TABLE_LABELS = {DocItemLabel.TABLE}
_FIGURE_LABELS = {DocItemLabel.PICTURE}


class DoclingParser(DocumentParser):
    """Parses PDF/DOCX/Markdown via Docling, preserving headings, tables and page numbers."""

    def __init__(self) -> None:
        # Instantiating DocumentConverter loads Docling's backend models, so it is
        # deferred until the first successful parse of a supported file type.
        self._converter: DocumentConverter | None = None

    def parse(self, file_path: str, document_id: str, filename: str) -> Document:
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        try:
            result = self._get_converter().convert(file_path)
            docling_document = result.document
        except Exception as exc:
            raise ParsingError(f"Failed to parse '{filename}': {exc}") from exc

        return self._to_domain_document(docling_document, document_id, filename)

    def _get_converter(self) -> DocumentConverter:
        if self._converter is None:
            self._converter = DocumentConverter()
        return self._converter

    def _to_domain_document(self, docling_document, document_id: str, filename: str) -> Document:
        sections: list[Section] = []
        current_section = Section(heading="", level=0)
        sections.append(current_section)

        for item, level in docling_document.iterate_items():
            label = getattr(item, "label", None)
            page_number = self._page_number(item)

            if label in _HEADER_LABELS:
                current_section = Section(heading=self._item_text(item), level=level)
                sections.append(current_section)
                continue

            if label in _TABLE_LABELS:
                current_section.elements.append(
                    Table(
                        text=self._table_text(item, docling_document),
                        page_number=page_number,
                        heading=current_section.heading or None,
                    )
                )
                continue

            if label in _FIGURE_LABELS:
                current_section.elements.append(
                    Figure(
                        caption=self._item_text(item) or None,
                        page_number=page_number,
                        heading=current_section.heading or None,
                    )
                )
                continue

            text = self._item_text(item)
            if not text:
                continue
            current_section.elements.append(
                Paragraph(text=text, page_number=page_number, heading=current_section.heading or None)
            )

        non_empty_sections = [section for section in sections if section.elements]
        return Document(document_id=document_id, filename=filename, sections=non_empty_sections)

    @staticmethod
    def _item_text(item) -> str:
        text = getattr(item, "text", None)
        return text.strip() if isinstance(text, str) else ""

    @staticmethod
    def _page_number(item) -> int | None:
        prov = getattr(item, "prov", None)
        if not prov:
            return None
        return getattr(prov[0], "page_no", None)

    @staticmethod
    def _table_text(item, docling_document) -> str:
        export = getattr(item, "export_to_markdown", None)
        if callable(export):
            try:
                return export(docling_document)
            except TypeError:
                return export()
        return DoclingParser._item_text(item)
