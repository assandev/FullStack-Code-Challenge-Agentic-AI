"""LLM-backed triage agent for first-contact bank support routing."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any, Protocol

from pydantic import ValidationError

from ..config import get_settings
from ..logging_utils import get_logger, log_json, log_phase
from ..schemas import TriageResult


class StructuredInvoker(Protocol):
    """Small protocol to simplify testing of the LangChain pipeline."""

    def invoke(self, input: dict[str, Any], config: Any | None = None, **kwargs: Any) -> Any:
        """Invoke the structured classifier."""


class TriageAgent:
    """Classifies incoming support queries for routing decisions."""

    def __init__(
        self,
        classifier: StructuredInvoker | None = None,
        prompt_path: Path | None = None,
    ) -> None:
        self._settings = get_settings()
        self._prompt_path = prompt_path or Path(__file__).resolve().parents[1] / "prompts" / "triage_prompt.txt"
        self._classifier = classifier
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8").strip()
        self._logger = get_logger("triage")

    def classify(self, query: str) -> TriageResult:
        """Run LLM triage and normalize the result into a safe contract."""

        log_phase(self._logger, "PHASE 1 | Triage Agent")
        self._logger.info("Preparing triage prompt")
        self._logger.info("Connected to %s", self._settings.ollama_model)
        self._logger.info("Waiting for Llama")
        try:
            result = self._get_classifier().invoke({"query": query})
            self._logger.info("Json returned")
            normalized = self._normalize_result(result)
            log_json(self._logger, "Triage result", normalized)
            return normalized
        except Exception as exc:  # pragma: no cover - exercised via tests using broad failures
            self._logger.warning("Triage fallback triggered: %s", type(exc).__name__)
            self._logger.warning("Triage exception detail: %s", str(exc))
            fallback = self._fallback_result(exc)
            log_json(self._logger, "Triage fallback result", fallback)
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
                    HumanMessagePromptTemplate.from_template("{query}"),
                ]
            )
            llm = ChatOllama(
                model=self._settings.ollama_model,
                base_url=self._settings.ollama_base_url,
                temperature=0,
            )
            self._classifier = prompt | llm
        return self._classifier

    def _normalize_result(self, raw_result: Any) -> TriageResult:
        if isinstance(raw_result, TriageResult):
            result = raw_result
        elif raw_content := self._extract_raw_content(raw_result):
            payload = self._extract_json_object(raw_content)
            if payload is None:
                self._logger.warning("Raw triage model response")
                self._logger.warning(raw_content)
                raise ValueError(f"Could not recover JSON from triage response: {raw_content}")
            result = TriageResult.model_validate(self._clamp_confidence(payload))
        elif isinstance(raw_result, Mapping):
            result = TriageResult.model_validate(self._clamp_confidence(dict(raw_result)))
        else:
            result = TriageResult.model_validate(raw_result)

        normalized = result.model_copy(
            update={
                "confidence": min(1.0, max(0.0, result.confidence)),
                "reason": result.reason.strip() or "The classification result did not include a reason.",
            }
        )

        if not normalized.is_bank_related:
            normalized = normalized.model_copy(update={"query_type": None, "needs_human": False})

        return normalized

    def _extract_raw_content(self, raw_message: Any) -> str | None:
        content = getattr(raw_message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            return "\n".join(part for part in text_parts if part).strip() or None
        return None

    def _extract_json_object(self, raw_content: str) -> dict[str, Any] | None:
        try:
            loaded = json.loads(raw_content)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        if match is None:
            return None

        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _clamp_confidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "confidence" in payload:
            try:
                payload["confidence"] = min(1.0, max(0.0, float(payload["confidence"])))
            except (TypeError, ValueError):
                pass
        return payload

    def _fallback_result(self, exc: Exception) -> TriageResult:
        error_name = type(exc).__name__
        if isinstance(exc, ValidationError):
            reason = "The triage output was malformed, so the request should be reviewed by a human."
        else:
            reason = f"Triage failed during {error_name}, so the request should be reviewed by a human."

        return TriageResult(
            is_bank_related=True,
            query_type=None,
            needs_human=True,
            confidence=0.0,
            reason=reason,
        )
