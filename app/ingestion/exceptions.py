"""Errors raised during document parsing and ingestion."""


class IngestionError(Exception):
    """Base class for ingestion failures mapped to HTTP error responses."""

    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFormatError(IngestionError):
    """Raised when the uploaded file extension or type is not supported."""


class EmptyDocumentError(IngestionError):
    """Raised when a file contains no extractable text."""


class FileTooLargeError(IngestionError):
    """Raised when an upload exceeds the configured size limit."""

    status_code = 413
