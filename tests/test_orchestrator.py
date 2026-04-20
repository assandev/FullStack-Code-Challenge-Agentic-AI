"""Tests for routing decisions across triage, support, and KB retrieval."""

from __future__ import annotations

import logging
import unittest

try:
    from bank_support_multi_agent.app.logging_utils import LOGGER_NAME
    from bank_support_multi_agent.app.orchestrator import SupportOrchestrator
    from bank_support_multi_agent.app.schemas import (
        EscalationAgentInput,
        EscalationResult,
        KBSearchResultItem,
        QueryType,
        SupportAgentInput,
        SupportResult,
        TriageResult,
    )
except ModuleNotFoundError:
    from app.logging_utils import LOGGER_NAME
    from app.orchestrator import SupportOrchestrator
    from app.schemas import (
        EscalationAgentInput,
        EscalationResult,
        KBSearchResultItem,
        QueryType,
        SupportAgentInput,
        SupportResult,
        TriageResult,
    )


class StubTriageAgent:
    def __init__(self, result: TriageResult):
        self.result = result

    def classify(self, query: str) -> TriageResult:
        return self.result


class StubSupportAgent:
    def __init__(self, responses: list[SupportResult]):
        self.responses = responses
        self.calls: list[SupportAgentInput] = []

    def handle(self, support_input: SupportAgentInput) -> SupportResult:
        self.calls.append(support_input)
        return self.responses.pop(0)


class StubKBTool:
    def __init__(self, results: list[KBSearchResultItem]):
        self.results = results
        self.calls: list[tuple[str, QueryType | None]] = []

    def search_bank_kb(
        self, query: str, query_type: QueryType | None, limit: int = 5
    ) -> list[KBSearchResultItem]:
        self.calls.append((query, query_type))
        return self.results


class StubEscalationAgent:
    def __init__(self, result: EscalationResult):
        self.result = result
        self.calls: list[EscalationAgentInput] = []

    def handle(self, escalation_input: EscalationAgentInput) -> EscalationResult:
        self.calls.append(escalation_input)
        return self.result


class OrchestratorTests(unittest.TestCase):
    def test_escalates_when_triage_requires_human(self) -> None:
        escalation_agent = StubEscalationAgent(
            EscalationResult(
                customer_message="I’m sorry, but this issue needs immediate review by a human support specialist.",
                handoff_summary="Customer reports suspicious transactions and requires fraud review.",
                priority="high",
                department="fraud_support",
                reason="Potential unauthorized transactions.",
            )
        )
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=True,
                    confidence=0.95,
                    reason="Fraud concern.",
                )
            ),
            escalation_agent=escalation_agent,
        )

        result = orchestrator.route("I think someone used my card.")

        self.assertEqual(result.next_agent, "escalation_agent")
        self.assertIsNone(result.support)
        self.assertIsNotNone(result.escalation)
        self.assertEqual(result.escalation.department, "fraud_support")

    def test_support_direct_response_path(self) -> None:
        support_agent = StubSupportAgent(
            [
                SupportResult(
                    action="respond",
                    response_message="Please reset your password and update the app.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=False,
                    confidence=0.88,
                    reason="The issue can be answered directly.",
                )
            ]
        )
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=True,
                    query_type="technical",
                    needs_human=False,
                    confidence=0.88,
                    reason="Login issue.",
                )
            ),
            support_agent=support_agent,
        )

        result = orchestrator.route("The app won't let me log in.")

        self.assertEqual(result.next_agent, "support_agent")
        self.assertIsNotNone(result.support)
        self.assertEqual(result.support.action, "respond")
        self.assertIsNone(result.tool_result)
        self.assertIsNone(result.escalation)

    def test_support_tool_path_followed_by_grounded_response(self) -> None:
        support_agent = StubSupportAgent(
            [
                SupportResult(
                    action="use_tool",
                    response_message=None,
                    tool_name="search_bank_kb",
                    tool_query="monthly maintenance fee explanation",
                    needs_human=False,
                    confidence=0.8,
                    reason="The issue needs grounded billing guidance.",
                ),
                SupportResult(
                    action="respond",
                    response_message="Monthly maintenance fees may apply when waiver conditions are not met.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=False,
                    confidence=0.89,
                    reason="The issue was resolved using billing guidance.",
                ),
            ]
        )
        kb_results = [
            KBSearchResultItem(
                id="billing_1",
                title="Monthly Maintenance Fees",
                category="billing",
                content="Accounts may incur a monthly maintenance fee if they do not meet minimum balance requirements.",
                confidence=0.94,
                source="billing.md",
            )
        ]
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=False,
                    confidence=0.88,
                    reason="Billing issue.",
                )
            ),
            support_agent=support_agent,
            kb_tool=StubKBTool(kb_results),
        )

        result = orchestrator.route("Why was I charged a monthly maintenance fee?")

        self.assertEqual(result.next_agent, "support_agent")
        self.assertIsNotNone(result.support)
        self.assertEqual(result.support.action, "respond")
        self.assertEqual(result.tool_result, kb_results)
        self.assertIsNone(result.escalation)

    def test_support_escalates_after_retrieval(self) -> None:
        escalation_agent = StubEscalationAgent(
            EscalationResult(
                customer_message="This request requires review by a human support specialist.",
                handoff_summary="Customer requests transaction reversal. Manual billing review required.",
                priority="high",
                department="billing_support",
                reason="Transaction reversal requires manual handling.",
            )
        )
        support_agent = StubSupportAgent(
            [
                SupportResult(
                    action="use_tool",
                    response_message=None,
                    tool_name="search_bank_kb",
                    tool_query="reverse this transaction policy",
                    needs_human=False,
                    confidence=0.82,
                    reason="Needs policy guidance.",
                ),
                SupportResult(
                    action="escalate",
                    response_message="This request requires review by a human support specialist.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=True,
                    confidence=0.92,
                    reason="Transaction reversals require human review.",
                ),
            ]
        )
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=False,
                    confidence=0.91,
                    reason="Transaction issue.",
                )
            ),
            support_agent=support_agent,
            kb_tool=StubKBTool([]),
            escalation_agent=escalation_agent,
        )

        result = orchestrator.route("Please reverse this transaction immediately.")

        self.assertEqual(result.next_agent, "escalation_agent")
        self.assertIsNotNone(result.support)
        self.assertEqual(result.support.action, "escalate")
        self.assertIsNotNone(result.escalation)
        self.assertEqual(result.escalation.department, "billing_support")

    def test_non_bank_queries_bypass_support(self) -> None:
        support_agent = StubSupportAgent([])
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=False,
                    query_type=None,
                    needs_human=False,
                    confidence=0.99,
                    reason="Not a banking question.",
                )
            ),
            support_agent=support_agent,
        )

        result = orchestrator.route("Tell me a joke.")

        self.assertIsNone(result.next_agent)
        self.assertIsNone(result.support)
        self.assertIsNone(result.escalation)
        self.assertEqual(support_agent.calls, [])

    def test_logs_route_flow_with_retrieval(self) -> None:
        support_agent = StubSupportAgent(
            [
                SupportResult(
                    action="use_tool",
                    response_message=None,
                    tool_name="search_bank_kb",
                    tool_query="monthly maintenance fee explanation",
                    needs_human=False,
                    confidence=0.8,
                    reason="The issue needs grounded billing guidance.",
                ),
                SupportResult(
                    action="respond",
                    response_message="Monthly maintenance fees may apply when waiver conditions are not met.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=False,
                    confidence=0.89,
                    reason="The issue was resolved using billing guidance.",
                ),
            ]
        )
        orchestrator = SupportOrchestrator(
            triage_agent=StubTriageAgent(
                TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=False,
                    confidence=0.88,
                    reason="Billing issue.",
                )
            ),
            support_agent=support_agent,
            kb_tool=StubKBTool(
                [
                    KBSearchResultItem(
                        id="billing_1",
                        title="Monthly Maintenance Fees",
                        category="billing",
                        content="Accounts may incur a monthly maintenance fee if they do not meet minimum balance requirements.",
                        confidence=0.94,
                        source="billing.md",
                    )
                ]
            ),
        )

        with self.assertLogs(f"{LOGGER_NAME}.orchestrator", level=logging.INFO) as captured:
            orchestrator.route("Why was I charged a monthly maintenance fee?")

        output = "\n".join(captured.output)
        self.assertIn("Route started for query", output)
        self.assertIn("Support requested tool execution", output)
        self.assertIn("Starting support final response phase", output)
        self.assertIn("Final routing decision: support_agent", output)


if __name__ == "__main__":
    unittest.main()
