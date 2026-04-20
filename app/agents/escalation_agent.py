"""LLM-backed escalation agent for final human handoff preparation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from ..config import get_settings
from ..logging_utils import get_logger, log_json, log_phase
from ..schemas import EscalationAgentInput, EscalationResult


class StructuredInvoker(Protocol):
    """Small protocol to simplify testing of the LangChain pipeline."""

    def invoke(self, input: dict[str, Any], config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the structured classifier."""


class EscalationAgent:
    """Creates the customer-facing escalation response and human handoff summary."""

    name = "escalation_agent"

    def __init__(
        self,
        classifier: StructuredInvoker | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = get_settings()
        self._prompt_path = prompt_path or Path(__file__).resolve().parents[1] / "prompts" / "escalation_prompt.txt"
        self._classifier = classifier
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8").strip()
        self._logger = get_logger("escalation")

    def handle(self, escalation_input: EscalationAgentInput) -> EscalationResult:
        """Run escalation handoff preparation and normalize the result."""

        log_phase(self._logger, "PHASE 5 | Escalation Agent")
        self._logger.info("Triggered by: %s", escalation_input.escalation_context.triggered_by)
        self._logger.info("Connected to %s", self._settings.ollama_model)
        self._logger.info("Waiting for Llama")
        try:
            result = self._get_classifier().invoke({"payload_json": escalation_input.model_dump_json(indent=2)})
            self._logger.info("Json returned")
            normalized = self._normalize_result(result)
            log_json(self._logger, "Escalation result", normalized)
            return normalized
        except Exception as exc:  # pragma: no cover - exercised in tests via broad failures
            self._logger.warning("Escalation fallback triggered: %s", type(exc).__name__)
            fallback = self._fallback_result(exc, escalation_input)
            log_json(self._logger, "Escalation fallback result", fallback)
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
            self._classifier = prompt | llm.with_structured_output(EscalationResult)
        return self._classifier

    def _normalize_result(self, raw_result: Any) -> EscalationResult:
        if isinstance(raw_result, EscalationResult):
            result = raw_result
        elif isinstance(raw_result, Mapping):
            result = EscalationResult.model_validate(dict(raw_result))
        else:
            result = EscalationResult.model_validate(raw_result)

        return result.model_copy(
            update={
                "customer_message": result.customer_message.strip(),
                "handoff_summary": result.handoff_summary.strip(),
                "reason": result.reason.strip(),
            }
        )

    def _fallback_result(self, exc: Exception, escalation_input: EscalationAgentInput) -> EscalationResult:
        reason = escalation_input.escalation_context.escalation_reason.strip()
        department = self._default_department(reason, escalation_input)
        priority = "high" if self._is_high_priority(reason, escalation_input) else "medium"

        if isinstance(exc, ValidationError):
            fallback_reason = "The escalation output was malformed, so a safe human handoff was prepared."
        else:
            fallback_reason = f"Escalation failed during {type(exc).__name__}, so a safe human handoff was prepared."

        return EscalationResult(
            customer_message="I’m sorry, but this issue needs to be reviewed by a human support specialist. Your case will be forwarded for further assistance.",
            handoff_summary=f"Customer needs human review. {reason}",
            priority=priority,
            department=department,
            reason=fallback_reason,
        )

    def _default_department(self, reason: str, escalation_input: EscalationAgentInput) -> str:
        text = " ".join(
            [
                escalation_input.user_query,
                reason,
                escalation_input.triage_result.reason,
                escalation_input.support_result.reason if escalation_input.support_result else "",
            ]
        ).lower()
        if "fraud" in text or "unauthorized" in text or "suspicious" in text:
            return "fraud_support"
        if escalation_input.triage_result.query_type == "technical":
            return "technical_support"
        if escalation_input.triage_result.query_type == "billing":
            return "billing_support"
        return "general_support"

    def _is_high_priority(self, reason: str, escalation_input: EscalationAgentInput) -> bool:
        text = " ".join(
            [
                escalation_input.user_query,
                reason,
                escalation_input.triage_result.reason,
                escalation_input.support_result.reason if escalation_input.support_result else "",
            ]
        ).lower()
        high_signals = ("fraud", "unauthorized", "suspicious", "blocked", "locked", "urgent", "reverse")
        return any(signal in text for signal in high_signals)
