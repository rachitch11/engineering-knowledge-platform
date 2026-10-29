"""Ephemeral per-visitor session storage for isolated RAG indexes and chat history."""

from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.generation.answer import AnswerService
from app.generation.chat_service import ChatService
from app.generation.models import ChatMessage
from app.generation.providers import LLMProvider, build_llm_provider
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.indexer import IndexService
from app.retrieval.reranker import Reranker
from app.retrieval.search import SearchService

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_SESSION_ID_PATTERN = uuid.UUID  # validates version 4 below


@dataclass
class SessionRecord:
    session_id: str
    root: Path
    settings: Settings
    index_service: IndexService
    search_service: SearchService
    chat_history: list[ChatMessage] = field(default_factory=list)
    chat_service: ChatService | None = None
    answer_service: AnswerService | None = None
    store_version: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock)


class SessionManager:
    """In-process registry of temporary session indexes (not shared across workers)."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_model: EmbeddingModel | None = None,
        reranker: Reranker | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_model = embedding_model
        self._reranker = reranker
        self._llm_provider = llm_provider
        self._sessions: dict[str, SessionRecord] = {}
        self._global_lock = threading.Lock()
        self._store_root = Path(settings.session_store_root)
        self._store_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_session_id(value: str) -> bool:
        try:
            parsed = uuid.UUID(value)
        except (ValueError, AttributeError, TypeError):
            return False
        return parsed.version == 4

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        root = self._store_root / session_id
        root.mkdir(parents=True, exist_ok=False)
        session_settings = self._settings.model_copy(
            update={
                "chroma_path": str(root / "chroma"),
                "bm25_index_path": str(root / "bm25_index.json"),
                "chroma_collection_name": f"session_{session_id.replace('-', '')[:16]}",
            }
        )
        index_service = IndexService.from_settings(
            session_settings,
            embedding_model=self._embedding_model,
        )
        search_service = SearchService(index_service, session_settings)
        record = SessionRecord(
            session_id=session_id,
            root=root,
            settings=session_settings,
            index_service=index_service,
            search_service=search_service,
            store_version=0,
        )
        with self._global_lock:
            self._sessions[session_id] = record
        logger.info("session_created session_id=%s", session_id)
        return session_id

    def touch(self, session_id: str) -> SessionRecord:
        if not self.is_valid_session_id(session_id):
            raise KeyError("invalid session id")
        with self._global_lock:
            record = self._sessions.get(session_id)
            if record is None:
                raise KeyError("unknown session")
        with record.lock:
            record.last_access = time.time()
        return record

    def get_record(self, session_id: str) -> SessionRecord:
        return self.touch(session_id)

    def get_index_service(self, session_id: str) -> IndexService:
        return self.get_record(session_id).index_service

    def get_search_service(self, session_id: str) -> SearchService:
        return self.get_record(session_id).search_service

    def get_chat_history(self, session_id: str) -> list[ChatMessage]:
        record = self.get_record(session_id)
        with record.lock:
            return list(record.chat_history)

    def append_chat_turn(
        self,
        session_id: str,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        record = self.get_record(session_id)
        with record.lock:
            record.chat_history.append(
                ChatMessage(role="user", content=user_message)
            )
            record.chat_history.append(
                ChatMessage(role="assistant", content=assistant_message)
            )

    def _rebuild_index(self, record: SessionRecord) -> None:
        record.store_version += 1
        store_dir = record.root / f"store_v{record.store_version}"
        store_dir.mkdir(parents=True, exist_ok=True)
        session_settings = record.settings.model_copy(
            update={
                "chroma_path": str(store_dir / "chroma"),
                "bm25_index_path": str(store_dir / "bm25_index.json"),
            }
        )
        record.settings = session_settings
        record.index_service = IndexService.from_settings(
            session_settings,
            embedding_model=self._embedding_model,
        )
        record.search_service = SearchService(
            record.index_service,
            session_settings,
        )

    def reset_session(self, session_id: str) -> None:
        record = self.get_record(session_id)
        with record.lock:
            record.chat_history.clear()
            record.chat_service = None
            record.answer_service = None
            self._rebuild_index(record)

    def destroy_session(self, session_id: str) -> None:
        if not self.is_valid_session_id(session_id):
            return
        with self._global_lock:
            record = self._sessions.pop(session_id, None)
        if record is None:
            return
        with record.lock:
            shutil.rmtree(record.root, ignore_errors=True)
        logger.info("session_destroyed session_id=%s", session_id)

    def cleanup_expired(self) -> int:
        ttl = self._settings.session_ttl_seconds
        now = time.time()
        expired: list[str] = []
        with self._global_lock:
            for session_id, record in self._sessions.items():
                if now - record.last_access > ttl:
                    expired.append(session_id)
        for session_id in expired:
            self.destroy_session(session_id)
        return len(expired)

    def shutdown_all(self) -> None:
        with self._global_lock:
            session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            self.destroy_session(session_id)

    def active_session_count(self) -> int:
        with self._global_lock:
            return len(self._sessions)

    def _build_chat_service(self, record: SessionRecord) -> ChatService:
        from app.generation.deps import build_reranker

        reranker = self._reranker or build_reranker(record.settings)
        llm = self._llm_provider or build_llm_provider(record.settings)
        return ChatService(
            record.search_service,
            record.settings,
            reranker=reranker,
            llm_provider=llm,
        )

    def _build_answer_service(self, record: SessionRecord) -> AnswerService:
        from app.generation.deps import build_reranker

        reranker = self._reranker or build_reranker(record.settings)
        llm = self._llm_provider or build_llm_provider(record.settings)
        return AnswerService(
            record.search_service,
            record.settings,
            reranker=reranker,
            llm_provider=llm,
        )

    def get_chat_service(self, session_id: str) -> ChatService:
        record = self.get_record(session_id)
        with record.lock:
            if record.chat_service is None:
                record.chat_service = self._build_chat_service(record)
            return record.chat_service

    def get_answer_service(self, session_id: str) -> AnswerService:
        record = self.get_record(session_id)
        with record.lock:
            if record.answer_service is None:
                record.answer_service = self._build_answer_service(record)
            return record.answer_service


def init_session_manager(
    app: FastAPI,
    settings: Settings,
    *,
    embedding_model: EmbeddingModel | None = None,
    reranker: Reranker | None = None,
    llm_provider: LLMProvider | None = None,
) -> SessionManager:
    manager = SessionManager(
        settings,
        embedding_model=embedding_model,
        reranker=reranker,
        llm_provider=llm_provider,
    )
    app.state.session_manager = manager  # type: ignore[attr-defined]
    return manager
