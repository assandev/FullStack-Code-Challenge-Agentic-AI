"""Tests for the escalation agent behavior and logging."""

from __future__ import annotations

import logging
import unittest

try:
    from bank_support_multi_agent.app.agents.escalation_agent import EscalationAgent
    from bank_support_multi_agent.app.logging_utils import LOGGER_NAME
    from bank_support_multi_agent.app.schemas import (
        EscalationAgentInput,
        EscalationContext,
        SupportResult,
        TriageResult,
    )
except ModuleNotFoundError:
    from app.agents.escalation_agent import EscalationAgent
    from app.logging_utils import LOGGER_NAME
    from app.schemas import EscalationAgentInput, EscalationContext, SupportResult, TriageResult


class StubClassifier:
    def __init__(self, response: object):
        self.response = response

    def invoke(self, input: dict[str, object], config: object | None = None, **kwargs: object) -> object:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class EscalationAgentTests(unittest.TestCase):
    def test_fraud_case_returns_high_priority(self) -> None:
        agent = EscalationAgent(
            classifier=StubClassifier(
                {
                    "customer_message": "I’m sorry, but this issue needs immediate review by a human support specialist.",
                    "handoff_summary": "Customer reports unauthorized transactions and needs urgent fraud review.",
                    "priority": "high",
                    "department": "fraud_support",
                    "reason": "Potential unauthorized transactions.",
                }
            )
        )

        result = agent.handle(
            EscalationAgentInput(
                user_query="I see three transactions I didn't make",
                triage_result=TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=True,
                    confidence=0.96,
                    reason="Potential unauthorized transactions.",
                ),
                support_result=None,
                escalation_context=EscalationContext(
                    triggered_by="triage",
                    escalation_reason="Potential fraud requires human review.",
                ),
            )
        )

        self.assertEqual(result.department, "fraud_support")
        self.assertEqual(result.priority, "high")

    def test_billing_case_returns_billing_support(self) -> None:
        agent = EscalationAgent(
            classifier=StubClassifier(
                {
                    "customer_message": "This request needs to be reviewed by a human support specialist.",
                    "handoff_summary": "Customer requests transaction reversal. Manual billing review required.",
                    "priority": "high",
                    "department": "billing_support",
                    "reason": "Transaction reversal requires manual handling.",
                }
            )
        )

        result = agent.handle(
            EscalationAgentInput(
                user_query="Please reverse this transaction immediately",
                triage_result=TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=False,
                    confidence=0.84,
                    reason="The user is asking about a transaction issue.",
                ),
                support_result=SupportResult(
                    action="escalate",
                    response_message="This request requires review by a human support specialist.",
                    tool_name=None,
                    tool_query=None,
                    needs_human=True,
                    confidence=0.93,
                    reason="Transaction reversal requests require manual review.",
                ),
                escalation_context=EscalationContext(
                    triggered_by="support",
                    escalation_reason="Transaction reversal requires manual handling.",
                ),
            )
        )

        self.assertEqual(result.department, "billing_support")

    def test_technical_case_returns_technical_support(self) -> None:
        agent = EscalationAgent(
            classifier=StubClassifier(
                {
                    "customer_message": "A human support specialist will review this issue.",
                    "handoff_summary": "Customer is locked out and recovery attempts failed.",
                    "priority": "high",
                    "department": "technical_support",
                    "reason": "Account access recovery failed.",
                }
            )
        )

        result = agent.handle(
            EscalationAgentInput(
                user_query="My account is locked and I still can't get back in",
                triage_result=TriageResult(
                    is_bank_related=True,
                    query_type="technical",
                    needs_human=True,
                    confidence=0.91,
                    reason="The user cannot recover account access.",
                ),
                support_result=None,
                escalation_context=EscalationContext(
                    triggered_by="triage",
                    escalation_reason="Account access recovery failed.",
                ),
            )
        )

        self.assertEqual(result.department, "technical_support")

    def test_malformed_output_falls_back_safely(self) -> None:
        agent = EscalationAgent(
            classifier=StubClassifier(
                {
                    "customer_message": "Needs handoff.",
                    "priority": "high",
                    "department": "fraud_support",
                    "reason": "Potential unauthorized transactions.",
                }
            )
        )

        result = agent.handle(
            EscalationAgentInput(
                user_query="I see suspicious activity",
                triage_result=TriageResult(
                    is_bank_related=True,
                    query_type="billing",
                    needs_human=True,
                    confidence=0.95,
                    reason="Potential unauthorized transactions.",
                ),
                support_result=None,
                escalation_context=EscalationContext(
                    triggered_by="triage",
                    escalation_reason="Potential fraud requires human review.",
                ),
            )
        )

        self.assertEqual(result.department, "fraud_support")
        self.assertEqual(result.priority, "high")

    def test_logs_escalation_phase(self) -> None:
        agent = EscalationAgent(
            classifier=StubClassifier(
                {
                    "customer_message": "This request needs review by a human support specialist.",
                    "handoff_summary": "Customer requests transaction reversal. Manual billing review required.",
                    "priority": "high",
                    "department": "billing_support",
                    "reason": "Transaction reversal requires manual handling.",
                }
            )
        )

        with self.assertLogs(f"{LOGGER_NAME}.escalation", level=logging.INFO) as captured:
            agent.handle(
                EscalationAgentInput(
                    user_query="Please reverse this transaction immediately",
                    triage_result=TriageResult(
                        is_bank_related=True,
                        query_type="billing",
                        needs_human=False,
                        confidence=0.84,
                        reason="The user is asking about a transaction issue.",
                    ),
                    support_result=SupportResult(
                        action="escalate",
                        response_message="This request requires review by a human support specialist.",
                        tool_name=None,
                        tool_query=None,
                        needs_human=True,
                        confidence=0.93,
                        reason="Transaction reversal requests require manual review.",
                    ),
                    escalation_context=EscalationContext(
                        triggered_by="support",
                        escalation_reason="Transaction reversal requires manual handling.",
                    ),
                )
            )

        output = "\n".join(captured.output)
        self.assertIn("PHASE 5 | Escalation Agent", output)
        self.assertIn("Triggered by: support", output)
        self.assertIn("Escalation result", output)


if __name__ == "__main__":
    unittest.main()
