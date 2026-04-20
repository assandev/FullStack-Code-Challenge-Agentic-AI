"""Minimal orchestration layer for routing bank support requests."""

from __future__ import annotations

from typing import Protocol

from .agents.escalation_agent import EscalationAgent
from .agents.support_agent import SupportAgent
from .agents.triage_agent import TriageAgent
from .logging_utils import get_logger
from .schemas import (
    EscalationAgentInput,
    EscalationContext,
    EscalationResult,
    KBSearchResultItem,
    QueryType,
    RouteResponse,
    SupportAgentInput,
    SupportResult,
    TriageResult,
)
from .tools.bank_kb_tool import BankKnowledgeBaseTool


class TriageClassifier(Protocol):
    """Minimal interface required by the orchestrator."""

    def classify(self, query: str) -> TriageResult:
        """Return the triage result for a user query."""
        ...


class SupportDecider(Protocol):
    """Minimal interface required by the orchestrator from the support agent."""

    def handle(self, support_input: SupportAgentInput) -> SupportResult:
        """Return the support decision for a routed query."""
        ...


class KnowledgeSearcher(Protocol):
    """Minimal interface required by the orchestrator from the KB tool."""

    def search_bank_kb(
        self, query: str, query_type: QueryType | None, limit: int = 5
    ) -> list[KBSearchResultItem]:
        """Return ranked KB matches for the query."""
        ...


class EscalationPreparer(Protocol):
    """Minimal interface required by the orchestrator from the escalation agent."""

    def handle(self, escalation_input: EscalationAgentInput) -> EscalationResult:
        """Return the escalation handoff package for a routed query."""
        ...


class SupportOrchestrator:
    """Routes queries based on triage output."""

    def __init__(
        self,
        triage_agent: TriageClassifier | None = None,
        support_agent: SupportDecider | None = None,
        kb_tool: KnowledgeSearcher | None = None,
        escalation_agent: EscalationPreparer | None = None,
    ) -> None:
        self.triage_agent = triage_agent or TriageAgent()
        self.support_agent = support_agent or SupportAgent()
        self.kb_tool = kb_tool or BankKnowledgeBaseTool()
        self.escalation_agent = escalation_agent or EscalationAgent()
        self._logger = get_logger("orchestrator")

    def route(self, query: str) -> RouteResponse:
        self._logger.info("Route started for query: %s", query)
        triage = self.triage_agent.classify(query)

        if triage.needs_human:
            self._logger.info("Triage short-circuited to escalation_agent")
            escalation = self.escalation_agent.handle(
                EscalationAgentInput(
                    user_query=query,
                    triage_result=triage,
                    support_result=None,
                    escalation_context=EscalationContext(triggered_by="triage", escalation_reason=triage.reason),
                )
            )
            return RouteResponse(
                triage=triage,
                support=None,
                tool_result=None,
                escalation=escalation,
                next_agent="escalation_agent",
            )

        if not triage.is_bank_related:
            self._logger.info("Non-bank query detected, skipping support flow")
            return RouteResponse(triage=triage, support=None, tool_result=None, escalation=None, next_agent=None)

        self._logger.info("Starting support decision phase")
        initial_input = SupportAgentInput(user_query=query, triage_result=triage)
        initial_support = self.support_agent.handle(initial_input)

        if initial_support.action == "use_tool":
            self._logger.info("Support requested tool execution: %s", initial_support.tool_name)
            tool_result = self.kb_tool.search_bank_kb(initial_support.tool_query or query, triage.query_type)
            self._logger.info("Starting support final response phase")
            final_input = initial_input.model_copy(update={"tool_result": tool_result})
            final_support = self.support_agent.handle(final_input)
            next_agent = "escalation_agent" if final_support.needs_human else "support_agent"
            escalation = None
            if final_support.needs_human:
                escalation = self.escalation_agent.handle(
                    EscalationAgentInput(
                        user_query=query,
                        triage_result=triage,
                        support_result=final_support,
                        escalation_context=EscalationContext(
                            triggered_by="support",
                            escalation_reason=final_support.reason,
                        ),
                    )
                )
            self._logger.info("Final routing decision: %s", next_agent)
            return RouteResponse(
                triage=triage,
                support=final_support,
                tool_result=tool_result,
                escalation=escalation,
                next_agent=next_agent,
            )

        next_agent = "escalation_agent" if initial_support.needs_human else "support_agent"
        escalation = None
        if initial_support.needs_human:
            escalation = self.escalation_agent.handle(
                EscalationAgentInput(
                    user_query=query,
                    triage_result=triage,
                    support_result=initial_support,
                    escalation_context=EscalationContext(
                        triggered_by="support",
                        escalation_reason=initial_support.reason,
                    ),
                )
            )
        self._logger.info("Final routing decision: %s", next_agent)
        return RouteResponse(
            triage=triage,
            support=initial_support,
            tool_result=None,
            escalation=escalation,
            next_agent=next_agent,
        )
