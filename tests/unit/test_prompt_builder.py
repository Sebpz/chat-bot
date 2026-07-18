from app.domain.chunk import Chunk, ScoredChunk
from app.services.prompt_builder import SYSTEM_PROMPT, build_context, build_user_prompt


def _scored(text: str, *, document_name="handbook.pdf", page_number=14, heading="Annual Leave", score=1.0):
    chunk = Chunk(
        chunk_id="c1",
        document_id="doc-1",
        document_name=document_name,
        page_number=page_number,
        heading=heading,
        text=text,
        token_count=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=score)


def test_build_context_includes_document_page_and_heading():
    context = build_context([_scored("Employees receive 20 days of leave.")])

    assert "Document: handbook.pdf" in context
    assert "Page: 14" in context
    assert "Section: Annual Leave" in context
    assert "Employees receive 20 days of leave." in context


def test_build_context_merges_multiple_chunks_in_order():
    chunks = [_scored("First chunk.", page_number=1), _scored("Second chunk.", page_number=2)]

    context = build_context(chunks)

    assert context.index("First chunk.") < context.index("Second chunk.")
    assert "---" in context


def test_build_context_respects_max_tokens_budget():
    long_chunks = [_scored(" ".join(f"word{i}" for i in range(500)), page_number=i) for i in range(10)]

    context = build_context(long_chunks, max_tokens=50)

    # At least the first chunk is always included, and the budget keeps later ones out.
    assert "word0" in context
    assert "Document: handbook.pdf" in context
    assert context.count("Document: handbook.pdf") < len(long_chunks)


def test_build_context_omits_missing_page_and_heading():
    chunk = _scored("No page or heading here.", page_number=None, heading=None)

    context = build_context([chunk])

    assert "Page:" not in context
    assert "Section:" not in context
    assert "No page or heading here." in context


def test_build_user_prompt_includes_question_and_context():
    prompt = build_user_prompt("How many leave days?", "Document: handbook.pdf\n\ncontext body")

    assert "How many leave days?" in prompt
    assert "context body" in prompt


def test_system_prompt_matches_spec_requirements():
    assert "ONLY" in SYSTEM_PROMPT
    assert "cite" in SYSTEM_PROMPT.lower()
