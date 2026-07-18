from app.core.config import get_settings
from app.domain.chunk import ScoredChunk
from app.utils.tokens import count_tokens

SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied context.\n\n"
    "If the answer cannot be found, clearly state that.\n\n"
    "Always cite the document and page number."
)


def _format_chunk(scored: ScoredChunk) -> str:
    chunk = scored.chunk
    lines = [f"Document: {chunk.document_name}"]
    if chunk.page_number is not None:
        lines.append(f"Page: {chunk.page_number}")
    if chunk.heading:
        lines.append(f"Section: {chunk.heading}")
    lines.append("")
    lines.append(chunk.text)
    return "\n".join(lines)


def build_context(chunks: list[ScoredChunk], max_tokens: int | None = None) -> str:
    """Merge reranked chunks into a single context block, capped at max_context_tokens."""
    settings = get_settings()
    limit = max_tokens if max_tokens is not None else settings.max_context_tokens

    blocks: list[str] = []
    total = 0
    for scored in chunks:
        block = _format_chunk(scored)
        block_tokens = count_tokens(block)
        if blocks and total + block_tokens > limit:
            break
        blocks.append(block)
        total += block_tokens
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, context: str) -> str:
    return f"Question: {question}\n\nContext:\n{context}"
