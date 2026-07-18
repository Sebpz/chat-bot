class AppError(Exception):
    """Base class for domain errors that the API layer maps to HTTPException."""


class UnsupportedFileTypeError(AppError):
    pass


class ParsingError(AppError):
    pass


class EmbeddingError(AppError):
    pass


class VectorStoreError(AppError):
    pass


class RerankingError(AppError):
    pass


class LLMError(AppError):
    pass


class DocumentNotFoundError(AppError):
    pass
