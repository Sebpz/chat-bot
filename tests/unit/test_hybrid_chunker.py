from app.domain.document import Document, Figure, Paragraph, Section, Table
from app.implementations.hybrid_chunker import HybridDocumentChunker


def make_chunker(chunk_size: int = 900, chunk_overlap: int = 150) -> HybridDocumentChunker:
    return HybridDocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunks_multiple_sections_with_heading_and_page_number():
    document = Document(
        document_id="doc-1",
        filename="handbook.pdf",
        sections=[
            Section(
                heading="Annual Leave",
                level=1,
                elements=[Paragraph(text="Employees receive 20 days of leave per year.", page_number=14)],
            ),
            Section(
                heading="Sick Leave",
                level=1,
                elements=[
                    Paragraph(text="Employees may take sick leave as needed.", page_number=15),
                    Table(text="| Type | Days |\n| Sick | 10 |", page_number=15),
                    Figure(caption="Sick leave policy diagram", page_number=15),
                ],
            ),
        ],
    )

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 2
    leave_chunk = next(c for c in chunks if c.heading == "Annual Leave")
    sick_chunk = next(c for c in chunks if c.heading == "Sick Leave")

    assert leave_chunk.page_number == 14
    assert "20 days" in leave_chunk.text
    assert leave_chunk.document_id == "doc-1"
    assert leave_chunk.document_name == "handbook.pdf"
    assert leave_chunk.token_count > 0

    assert sick_chunk.page_number == 15
    assert "sick leave as needed" in sick_chunk.text
    assert "Sick | 10" in sick_chunk.text
    assert "Sick leave policy diagram" in sick_chunk.text


def test_long_section_produces_multiple_overlapping_chunks():
    long_text = " ".join(f"word{i}" for i in range(2000))
    document = Document(
        document_id="doc-2",
        filename="long.pdf",
        sections=[Section(heading="Body", elements=[Paragraph(text=long_text, page_number=1)])],
    )

    chunks = make_chunker(chunk_size=100, chunk_overlap=20).chunk(document)

    assert len(chunks) > 1
    for c in chunks:
        assert c.heading == "Body"
        assert c.page_number == 1
        assert c.token_count <= 100

    # Overlap: the tail of one chunk should reappear at the head of the next.
    first_tail_words = chunks[0].text.split()[-5:]
    second_words = chunks[1].text.split()
    assert any(word in second_words for word in first_tail_words)


def test_empty_sections_list_falls_back_to_no_chunks():
    document = Document(document_id="doc-3", filename="empty.pdf", sections=[])

    chunks = make_chunker().chunk(document)

    assert chunks == []


def test_no_empty_chunks_are_ever_produced():
    document = Document(
        document_id="doc-4",
        filename="mixed.pdf",
        sections=[
            Section(heading="Blank", elements=[Paragraph(text="   ", page_number=1)]),
            Section(heading="Real", elements=[Paragraph(text="Actual content here.", page_number=2)]),
        ],
    )

    chunks = make_chunker().chunk(document)

    assert len(chunks) == 1
    assert chunks[0].heading == "Real"
    for c in chunks:
        assert c.text.strip() != ""
