from abc import ABC, abstractmethod

from app.domain.document import Document


class DocumentParser(ABC):
    """Parses a source file (PDF, DOCX, Markdown, ...) into a structured Document."""

    @abstractmethod
    def parse(self, file_path: str, document_id: str, filename: str) -> Document:
        """Parse the file at file_path into a Document.

        Raises ParsingError (see app.utils.errors) on unsupported or corrupted input.
        """
        raise NotImplementedError
