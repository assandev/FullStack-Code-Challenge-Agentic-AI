"""LLM-backed support agent for post-triage bank support decisions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ..config import get_settings
from ..logging_utils import get_logger, log_json, log_phase
from ..schemas import SupportAgentInput, SupportResult


class StructuredInvoker(Protocol):
    """Small protocol to simplify testing of the LangChain pipeline."""

    def invoke(self, input: dict[str, Any], config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the structured classifier."""


class SupportAgent:
    """Decides whether to answer, retrieve knowledge, or escalate."""

    name = "support_agent"

    def __init__(
        self,
        classifier: StructuredInvoker | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = get_settings()
        self._prompt_path = prompt_path or Path(__file__).resolve().parents[1] / "prompts" / "support_prompt.txt"
        self._classifier = classifier
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8").strip()
        self._logger = get_logger("support")

    def handle(self, support_input: SupportAgentInput) -> SupportResult:
        """Run support reasoning and normalize the result into a safe contract."""

        phase_title = (
            "PHASE 2 | Support Agent - Decision"
            if support_input.tool_result is None
            else "PHASE 4 | Support Agent - Final Response"
        )
        log_phase(self._logger, phase_title)
        self._logger.info("Tool results present: %s", support_input.tool_result is not None)
        self._logger.info("Connected to %s", self._settings.ollama_model)
        self._logger.info("Waiting for Llama")
        try:
            result = self._get_classifier().invoke({"payload_json": support_input.model_dump_json(indent=2)})
            self._logger.info("Json returned")
            normalized = self._normalize_result(result, support_input)
            log_json(self._logger, "Support result", normalized)
            return normalized
        except Exception as exc:  # pragma: no cover - exercised in tests via broad failures
            self._logger.warning("Support fallback triggered: %s", type(exc).__name__)
            fallback = self._fallback_result(exc)
            log_json(self._logger, "Support fallback result", fallback)
            return fallback

    def _get_classifier(self) -> StructuredInvoker:
        if self._classifier is None:
            from langchain_core.messages import SystemMessage
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.prompts.chat import HumanMessagePromptTemplate
            from langchain_ollama import ChatOllama

            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self._system_prompt),
                    HumanMessagePromptTemplate.from_template("{payload_json}"),
                ]
            )
            llm = ChatOllama(
                model=self._settings.ollama_model,
                base_url=self._settings.ollama_base_url,
                temperature=0,
            )
            self._classifier = prompt | llm.with_structured_output(SupportResult)
        return self._classifier

    def _normalize_result(self, raw_result: Any, support_input: SupportAgentInput) -> SupportResult:
        if isinstance(raw_result, SupportResult):
            result = raw_result
        elif isinstance(raw_result, Mapping):
            payload = dict(raw_result)
            if "confidence" in payload:
                try:
                    payload["confidence"] = min(1.0, max(0.0, float(payload["confidence"])))
                except (TypeError, ValueError):
                    pass
            result = SupportResult.model_validate(payload)
        else:
            result = SupportResult.model_validate(raw_result)

        normalized = result.model_copy(
            update={
                "confidence": min(1.0, max(0.0, result.confidence)),
                "reason": result.reason.strip() or "The support decision did not include a reason.",
            }
        )

        if normalized.action == "use_tool":
            tool_query = (normalized.tool_query or "").strip()
            if support_input.tool_result is not None:
                return SupportResult(
                    action="escalate",
                    response_message="I wasn't able to safely resolve this with the available information, so a human support specialist should review it.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=True,
                    confidence=0.0,
                    reason="Support requested another retrieval pass after tool results were already provided.",
                )
            return normalized.model_copy(
                update={
                    "response_message": None,
                    "tool_name": "search_bank_kb",
                    "tool_query": tool_query or "bank support guidance",
                    "needs_human": False,
                }
            )

        if normalized.action == "respond":
            response_message = (normalized.response_message or "").strip()
            if not response_message:
                return self._fallback_result(ValidationError)
            return normalized.model_copy(
                update={
                    "response_message": response_message,
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                }
            )

        response_message = (normalized.response_message or "").strip()
        return normalized.model_copy(
            update={
                "response_message": response_message or "This request requires review by a human support specialist.",
                "tool_name": None,
                "tool_query": None,
                "needs_human": True,
            }
        )

    def _fallback_result(self, exc: Exception | type[Exception]) -> SupportResult:
        error_name = exc.__name__ if isinstance(exc, type) else type(exc).__name__
        if exc is ValidationError or isinstance(exc, ValidationError):
            reason = "The support output was malformed, so the request should be reviewed by a human."
        else:
            reason = f"Support failed during {error_name}, so the request should be reviewed by a human."

        return SupportResult(
            action="escalate",
            response_message="This request requires review by a human support specialist.",
            tool_name=None,
            tool_query=None,
            needs_human=True,
            confidence=0.0,
            reason=reason,
        )
