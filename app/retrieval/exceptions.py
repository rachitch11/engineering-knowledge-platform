"""Errors raised during indexing and retrieval."""


class RetrievalError(Exception):
    status_code: int = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class EmptyQueryError(RetrievalError):
    """Raised when a search query is blank after normalization."""


class DocumentNotFoundError(RetrievalError):
    status_code = 404


class IndexingError(RetrievalError):
    status_code = 500


class EmptyIndexError(RetrievalError):
    """Raised when search is attempted against an empty index."""
