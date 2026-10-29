"""Conversation-aware chat orchestration with intent routing and timing."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from app.core.config import Settings
from app.core.timing import PipelineTimer
from app.generation.answer import (
    build_citations,
    has_sufficient_evidence,
    hits_to_evidence,
)
from app.generation.conversation import (
    build_conversation_context,
    rewrite_retrieval_query,
    trim_history,
)
from app.generation.exceptions import GenerationError, LLMProviderError
from app.generation.identity_responses import (
    application_identity_response,
    configured_llm_identity,
    model_identity_response,
)
from app.generation.intent import (
    MessageIntent,
    ResponseMode,
    capabilities_response,
    classify_intent,
    greeting_response,
    intent_to_response_mode,
    is_standalone_general_question,
    normalize_intent_text,
)
from app.generation.models import ChatMessage, ChatResponse
from app.generation.providers import LLMProvider, build_llm_provider
from app.retrieval.reranker import Reranker
from app.retrieval.search import SearchService

logger = logging.getLogger(__name__)

_DOCUMENT_MISSING_MESSAGE = (
    "I searched your indexed documents but could not find information that answers "
    "this question. The detail may not be present in the files you indexed."
)


class ChatService:
    """Route intents, run retrieval when needed, and produce chat responses."""

    def __init__(
        self,
        search_service: SearchService,
        settings: Settings,
        *,
        reranker: Reranker | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self._search = search_service
        self._settings = settings
        self._reranker = reranker
        self._llm = llm_provider

    def chat(
        self,
        message: str,
        history: list[ChatMessage],
        *,
        strategy: str = "hybrid",
        top_k: int | None = None,
        candidate_k: int | None = None,
        rerank: bool | None = None,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> ChatResponse:
        timer = PipelineTimer()
        trimmed = trim_history(history, self._settings.max_conversation_messages)
        indexed_count = self._indexed_chunk_count()

        with timer.stage("intent"):
            intent = classify_intent(
                message,
                trimmed,
                indexed_chunk_count=indexed_count,
            )
        response_mode = intent_to_response_mode(intent)
        retrieval_invoked = False
        response_path = "static"

        if intent == MessageIntent.GREETING:
            response = self._static_response(
                message,
                greeting_response(),
                intent=intent,
                response_mode=response_mode,
                retrieval_query=message,
                timer=timer,
            )
            self._log_route(message, intent, response_mode, retrieval_invoked, response_path, timer)
            return response

        if intent == MessageIntent.APPLICATION_IDENTITY:
            response = self._static_response(
                message,
                application_identity_response(self._settings),
                intent=intent,
                response_mode=response_mode,
                retrieval_query=message,
                timer=timer,
            )
            self._log_route(message, intent, response_mode, retrieval_invoked, response_path, timer)
            return response

        if intent == MessageIntent.MODEL_IDENTITY:
            response = self._static_response(
                message,
                model_identity_response(self._settings),
                intent=intent,
                response_mode=response_mode,
                retrieval_query=message,
                timer=timer,
            )
            self._log_route(message, intent, response_mode, retrieval_invoked, response_path, timer)
            return response

        if intent == MessageIntent.CAPABILITIES:
            response = self._static_response(
                message,
                capabilities_response(),
                intent=intent,
                response_mode=response_mode,
                retrieval_query=message,
                timer=timer,
            )
            self._log_route(message, intent, response_mode, retrieval_invoked, response_path, timer)
            return response

        conversation_context = build_conversation_context(
            trimmed,
            max_messages=self._settings.max_conversation_messages,
            max_chars=self._settings.max_conversation_context_chars,
        )
        context_included = bool(conversation_context)

        if intent == MessageIntent.GENERAL_CHAT or is_standalone_general_question(
            message
        ):
            with timer.stage("generation"):
                answer_text = self._generate_general(
                    message,
                    conversation_context=conversation_context,
                    timer=timer,
                )
            response = self._build_chat_response(
                message=message,
                answer=answer_text,
                intent=MessageIntent.GENERAL_CHAT,
                response_mode=ResponseMode.GENERAL,
                retrieval_query=message,
                grounded=False,
                sufficient_evidence=False,
                citations=[],
                evidence=[],
                strategy=strategy,
                candidate_count=0,
                rerank_applied=False,
                timer=timer,
                conversation_context_included=context_included,
            )
            response_path = "general_llm"
            self._log_route(message, intent, ResponseMode.GENERAL, retrieval_invoked, response_path, timer)
            return response

        with timer.stage("query_rewrite"):
            retrieval_query = rewrite_retrieval_query(message, trimmed)

        retrieval_invoked = True
        if intent == MessageIntent.MIXED:
            response_path = "mixed_rag"
            response = self._mixed_response(
                message,
                retrieval_query=retrieval_query,
                conversation_context=conversation_context,
                context_included=context_included,
                intent=intent,
                timer=timer,
                strategy=strategy,
                top_k=top_k,
                candidate_k=candidate_k,
                rerank=rerank,
                document_id=document_id,
                file_type=file_type,
            )
        else:
            response_path = "document_rag"
            response = self._grounded_response(
                message,
                retrieval_query=retrieval_query,
                conversation_context=conversation_context,
                context_included=context_included,
                intent=intent,
                timer=timer,
                strategy=strategy,
                top_k=top_k,
                candidate_k=candidate_k,
                rerank=rerank,
                document_id=document_id,
                file_type=file_type,
            )
        self._log_route(message, intent, intent_to_response_mode(intent), retrieval_invoked, response_path, timer)
        return response

    def iter_chat_sse(self, response: ChatResponse) -> Iterator[str]:
        """Yield SSE ``data:`` lines for an already-built chat response.

        Call ``chat()`` before returning ``StreamingResponse`` so LLM and retrieval
        errors become normal HTTP responses instead of failing after headers are sent.
        """
        try:
            answer = response.answer
            chunk_size = 24
            for index in range(0, len(answer), chunk_size):
                piece = answer[index : index + chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'text': piece})}\n\n"

            payload = response.model_dump()
            yield f"data: {json.dumps({'type': 'done', 'response': payload})}\n\n"
        except Exception:
            logger.exception(
                "chat_sse_stream_failed after response start; terminating stream"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "detail": "Stream failed while sending the chat response.",
                    }
                )
                + "\n\n"
            )

    def _static_response(
        self,
        message: str,
        answer: str,
        *,
        intent: MessageIntent,
        response_mode: ResponseMode,
        retrieval_query: str,
        timer: PipelineTimer,
    ) -> ChatResponse:
        return self._build_chat_response(
            message=message,
            answer=answer,
            intent=intent,
            response_mode=response_mode,
            retrieval_query=retrieval_query,
            grounded=False,
            sufficient_evidence=False,
            citations=[],
            evidence=[],
            strategy="hybrid",
            candidate_count=0,
            rerank_applied=False,
            timer=timer,
            conversation_context_included=False,
        )

    def _grounded_response(
        self,
        message: str,
        *,
        retrieval_query: str,
        conversation_context: str,
        context_included: bool,
        intent: MessageIntent,
        timer: PipelineTimer,
        strategy: str,
        top_k: int | None,
        candidate_k: int | None,
        rerank: bool | None,
        document_id: str | None,
        file_type: str | None,
    ) -> ChatResponse:
        candidate_limit = candidate_k or self._settings.rerank_candidate_k
        final_k = top_k or self._settings.rerank_top_k
        use_rerank = self._should_rerank(message, rerank)

        with timer.stage("retrieval"):
            retrieval = self._search.search(
                retrieval_query,
                top_k=candidate_limit,
                strategy=strategy,  # type: ignore[arg-type]
                document_id=document_id,
                file_type=file_type,  # type: ignore[arg-type]
                timer=timer,
            )

        hits = retrieval.results
        rerank_applied = False
        if use_rerank and hits and self._reranker is not None:
            with timer.stage("rerank"):
                ranked_hits = self._reranker.rerank(message, hits, top_k=final_k)
            rerank_applied = True
        else:
            ranked_hits = hits[:final_k]

        with timer.stage("evidence_check"):
            sufficient = has_sufficient_evidence(
                message,
                hits,
                min_dense_score=self._settings.evidence_min_dense_score,
            )

        if not sufficient:
            return self._build_chat_response(
                message=message,
                answer=_DOCUMENT_MISSING_MESSAGE,
                intent=intent,
                response_mode=ResponseMode.DOCUMENT,
                retrieval_query=retrieval_query,
                grounded=True,
                sufficient_evidence=False,
                citations=[],
                evidence=hits_to_evidence(ranked_hits),
                strategy=retrieval.strategy,
                candidate_count=len(retrieval.results),
                rerank_applied=rerank_applied,
                timer=timer,
                conversation_context_included=context_included,
            )

        evidence = hits_to_evidence(ranked_hits[: self._settings.max_evidence_chunks])
        llm = self._resolve_llm()

        try:
            with timer.stage("generation"):
                with timer.stage("prompt_preparation"):
                    pass
                with timer.stage("generation.llm"):
                    generated = llm.generate_answer(
                        message,
                        evidence,
                        timeout_seconds=self._settings.llm_timeout_seconds,
                        max_context_chars=self._settings.max_context_chars,
                        conversation_context=conversation_context,
                        num_predict=self._settings.llm_max_output_tokens,
                    )
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Answer generation failed: {exc}") from exc

        citations = build_citations(evidence, generated.citation_ids)
        if not generated.answer.strip():
            return self._build_chat_response(
                message=message,
                answer=_DOCUMENT_MISSING_MESSAGE,
                intent=intent,
                response_mode=ResponseMode.DOCUMENT,
                retrieval_query=retrieval_query,
                grounded=True,
                sufficient_evidence=False,
                citations=[],
                evidence=evidence,
                strategy=retrieval.strategy,
                candidate_count=len(retrieval.results),
                rerank_applied=rerank_applied,
                timer=timer,
                conversation_context_included=context_included,
            )

        return self._build_chat_response(
            message=message,
            answer=generated.answer,
            intent=intent,
            response_mode=ResponseMode.DOCUMENT,
            retrieval_query=retrieval_query,
            grounded=True,
            sufficient_evidence=True,
            citations=citations,
            evidence=evidence,
            strategy=retrieval.strategy,
            candidate_count=len(retrieval.results),
            rerank_applied=rerank_applied,
            timer=timer,
            conversation_context_included=context_included,
        )

    def _mixed_response(
        self,
        message: str,
        *,
        retrieval_query: str,
        conversation_context: str,
        context_included: bool,
        intent: MessageIntent,
        timer: PipelineTimer,
        strategy: str,
        top_k: int | None,
        candidate_k: int | None,
        rerank: bool | None,
        document_id: str | None,
        file_type: str | None,
    ) -> ChatResponse:
        candidate_limit = candidate_k or self._settings.rerank_candidate_k
        final_k = top_k or self._settings.rerank_top_k
        use_rerank = self._should_rerank(message, rerank)

        with timer.stage("retrieval"):
            retrieval = self._search.search(
                retrieval_query,
                top_k=candidate_limit,
                strategy=strategy,  # type: ignore[arg-type]
                document_id=document_id,
                file_type=file_type,  # type: ignore[arg-type]
                timer=timer,
            )

        hits = retrieval.results
        rerank_applied = False
        if use_rerank and hits and self._reranker is not None:
            with timer.stage("rerank"):
                ranked_hits = self._reranker.rerank(message, hits, top_k=final_k)
            rerank_applied = True
        else:
            ranked_hits = hits[:final_k]

        evidence = hits_to_evidence(ranked_hits[: self._settings.max_evidence_chunks])
        llm = self._resolve_llm()
        generate_mixed = getattr(llm, "generate_mixed", None)

        try:
            with timer.stage("generation"):
                with timer.stage("prompt_preparation"):
                    pass
                with timer.stage("generation.llm"):
                    if callable(generate_mixed):
                        generated = generate_mixed(
                            message,
                            evidence,
                            timeout_seconds=self._settings.llm_timeout_seconds,
                            max_context_chars=self._settings.max_context_chars,
                            conversation_context=conversation_context,
                            num_predict=self._settings.llm_max_output_tokens,
                        )
                    else:
                        generated = llm.generate_answer(
                            message,
                            evidence,
                            timeout_seconds=self._settings.llm_timeout_seconds,
                            max_context_chars=self._settings.max_context_chars,
                            conversation_context=conversation_context,
                            num_predict=self._settings.llm_max_output_tokens,
                        )
        except LLMProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Mixed answer generation failed: {exc}") from exc

        citations = build_citations(evidence, generated.citation_ids)
        sufficient = bool(generated.citation_ids) and bool(evidence)
        return self._build_chat_response(
            message=message,
            answer=generated.answer,
            intent=intent,
            response_mode=ResponseMode.MIXED,
            retrieval_query=retrieval_query,
            grounded=True,
            sufficient_evidence=sufficient,
            citations=citations,
            evidence=evidence,
            strategy=retrieval.strategy,
            candidate_count=len(retrieval.results),
            rerank_applied=rerank_applied,
            timer=timer,
            conversation_context_included=context_included,
        )

    def _generate_general(
        self,
        message: str,
        *,
        conversation_context: str,
        timer: PipelineTimer,
    ) -> str:
        llm = self._resolve_llm()
        generate = getattr(llm, "generate_conversational", None)
        if callable(generate):
            with timer.stage("prompt_preparation"):
                pass
            with timer.stage("generation.llm"):
                return str(
                    generate(
                        message,
                        conversation_context=conversation_context,
                        timeout_seconds=self._settings.llm_timeout_seconds,
                        max_context_chars=self._settings.max_conversation_context_chars,
                        num_predict=self._settings.llm_max_output_tokens,
                    )
                )
        raise LLMProviderError(
            "Configured LLM provider does not support general conversational answers."
        )

    def _build_chat_response(
        self,
        *,
        message: str,
        answer: str,
        intent: MessageIntent,
        response_mode: ResponseMode,
        retrieval_query: str,
        grounded: bool,
        sufficient_evidence: bool,
        citations: list,
        evidence: list,
        strategy: str,
        candidate_count: int,
        rerank_applied: bool,
        timer: PipelineTimer,
        conversation_context_included: bool,
    ) -> ChatResponse:
        return ChatResponse(
            question=message,
            answer=answer,
            sufficient_evidence=sufficient_evidence,
            citations=citations,
            evidence=evidence,
            retrieval_strategy=strategy,  # type: ignore[arg-type]
            candidate_count=candidate_count,
            rerank_applied=rerank_applied,
            intent=intent.value,
            response_mode=response_mode.value,
            retrieval_query=retrieval_query,
            grounded=grounded,
            timings_ms=timer.as_dict(),
            conversation_context_included=conversation_context_included,
        )

    def _should_rerank(self, message: str, rerank: bool | None) -> bool:
        if rerank is False:
            return False
        if not self._settings.rerank_enabled:
            return False
        if rerank is True:
            return self._reranker is not None
        if self._settings.rerank_skip_short_queries:
            if len(message.split()) <= self._settings.rerank_skip_max_words:
                return False
        return self._reranker is not None

    def _indexed_chunk_count(self) -> int:
        try:
            return self._search._index.chroma.count()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            return 0

    def _resolve_llm(self) -> LLMProvider:
        if self._llm is not None:
            return self._llm
        self._llm = build_llm_provider(self._settings)
        return self._llm

    def _log_route(
        self,
        message: str,
        intent: MessageIntent,
        response_mode: ResponseMode,
        retrieval_invoked: bool,
        response_path: str,
        timer: PipelineTimer,
    ) -> None:
        provider, model = configured_llm_identity(self._settings)
        normalized = normalize_intent_text(message)
        if self._settings.log_pipeline_timings:
            logger.info(
                "chat_route normalized=%r intent=%s mode=%s retrieval=%s "
                "provider=%s model=%s path=%s timings_ms=%s",
                normalized,
                intent.value,
                response_mode.value,
                retrieval_invoked,
                provider,
                model or "unknown",
                response_path,
                timer.as_dict(),
            )

    def _log_timings(
        self,
        message: str,
        intent: MessageIntent,
        timer: PipelineTimer,
    ) -> None:
        _ = (message, intent, timer)
