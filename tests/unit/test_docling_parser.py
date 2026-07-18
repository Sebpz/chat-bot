from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.domain.document import Figure, Paragraph, Table
from app.utils.errors import ParsingError, UnsupportedFileTypeError


def _make_item(label, text="", page_no=None, export_to_markdown=None):
    prov = [SimpleNamespace(page_no=page_no)] if page_no is not None else []
    item = SimpleNamespace(label=label, text=text, prov=prov)
    if export_to_markdown is not None:
        item.export_to_markdown = export_to_markdown
    return item


def test_parse_maps_headings_paragraphs_tables_and_figures(tmp_path):
    from docling_core.types.doc import DocItemLabel

    from app.implementations.docling_parser import DoclingParser

    heading_item = _make_item(DocItemLabel.SECTION_HEADER, text="Annual Leave", page_no=14)
    paragraph_item = _make_item(DocItemLabel.TEXT, text="Employees receive 20 days.", page_no=14)
    table_item = _make_item(
        DocItemLabel.TABLE,
        page_no=15,
        export_to_markdown=lambda doc: "| Type | Days |\n| --- | --- |\n| Annual | 20 |",
    )
    figure_item = _make_item(DocItemLabel.PICTURE, text="Leave accrual chart", page_no=15)

    docling_document = MagicMock()
    docling_document.iterate_items.return_value = [
        (heading_item, 1),
        (paragraph_item, 1),
        (table_item, 1),
        (figure_item, 1),
    ]

    fake_result = SimpleNamespace(document=docling_document)
    file_path = tmp_path / "handbook.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake")

    with patch("app.implementations.docling_parser.DocumentConverter") as mock_converter_cls:
        mock_converter_cls.return_value.convert.return_value = fake_result
        parser = DoclingParser()
        document = parser.parse(str(file_path), document_id="doc-1", filename="handbook.pdf")

    mock_converter_cls.return_value.convert.assert_called_once_with(str(file_path))

    assert document.document_id == "doc-1"
    assert document.filename == "handbook.pdf"
    assert len(document.sections) == 1

    section = document.sections[0]
    assert section.heading == "Annual Leave"
    assert len(section.elements) == 3

    paragraph = section.elements[0]
    assert isinstance(paragraph, Paragraph)
    assert paragraph.text == "Employees receive 20 days."
    assert paragraph.page_number == 14
    assert paragraph.heading == "Annual Leave"

    table = section.elements[1]
    assert isinstance(table, Table)
    assert "Annual | 20" in table.text
    assert table.page_number == 15

    figure = section.elements[2]
    assert isinstance(figure, Figure)
    assert figure.caption == "Leave accrual chart"
    assert figure.page_number == 15


def test_parse_places_content_before_first_heading_in_untitled_section(tmp_path):
    from docling_core.types.doc import DocItemLabel

    from app.implementations.docling_parser import DoclingParser

    intro_item = _make_item(DocItemLabel.TEXT, text="Welcome to the handbook.", page_no=1)

    docling_document = MagicMock()
    docling_document.iterate_items.return_value = [(intro_item, 0)]
    fake_result = SimpleNamespace(document=docling_document)

    file_path = tmp_path / "handbook.md"
    file_path.write_text("Welcome to the handbook.")

    with patch("app.implementations.docling_parser.DocumentConverter") as mock_converter_cls:
        mock_converter_cls.return_value.convert.return_value = fake_result
        parser = DoclingParser()
        document = parser.parse(str(file_path), document_id="doc-2", filename="handbook.md")

    assert len(document.sections) == 1
    assert document.sections[0].heading == ""
    assert document.sections[0].elements[0].text == "Welcome to the handbook."


def test_parse_rejects_unsupported_extension(tmp_path):
    from app.implementations.docling_parser import DoclingParser

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello")

    parser = DoclingParser()
    with pytest.raises(UnsupportedFileTypeError):
        parser.parse(str(file_path), document_id="doc-3", filename="notes.txt")


def test_parse_wraps_docling_failures_as_parsing_error(tmp_path):
    from app.implementations.docling_parser import DoclingParser

    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"not a real pdf")

    with patch("app.implementations.docling_parser.DocumentConverter") as mock_converter_cls:
        mock_converter_cls.return_value.convert.side_effect = RuntimeError("corrupt stream")
        parser = DoclingParser()
        with pytest.raises(ParsingError):
            parser.parse(str(file_path), document_id="doc-4", filename="broken.pdf")
