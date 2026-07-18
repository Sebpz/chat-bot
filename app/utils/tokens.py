import tiktoken

_ENCODING_NAME = "cl100k_base"
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def split_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive/sliding-window token splitter used as the chunking fallback."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    encoding = _get_encoding()
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(encoding.decode(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks
