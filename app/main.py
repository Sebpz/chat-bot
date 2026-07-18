from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.logging import configure_logging
from app.utils.errors import (
    AppError,
    DocumentNotFoundError,
    EmbeddingError,
    LLMError,
    ParsingError,
    RerankingError,
    UnsupportedFileTypeError,
    VectorStoreError,
)

_STATUS_BY_ERROR: list[tuple[type[AppError], int]] = [
    (UnsupportedFileTypeError, 400),
    (DocumentNotFoundError, 404),
    (ParsingError, 422),
    (EmbeddingError, 502),
    (RerankingError, 502),
    (VectorStoreError, 503),
    (LLMError, 504),
]


def _status_for(exc: AppError) -> int:
    for error_type, status_code in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return status_code
    return 500


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(title="AI Document Chat Backend", version="0.1.0")
    app.include_router(router)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(exc), content={"detail": str(exc)})

    return app


app = create_app()
