"""Tests for the support agent and KB search behavior."""

from __future__ import annotations

import logging
import unittest

try:
    from bank_support_multi_agent.app.agents.support_agent import SupportAgent
    from bank_support_multi_agent.app.logging_utils import LOGGER_NAME
    from bank_support_multi_agent.app.schemas import KBSearchResultItem, SupportAgentInput, TriageResult
    from bank_support_multi_agent.app.tools.bank_kb_tool import BankKnowledgeBaseTool
except ModuleNotFoundError:
    from app.agents.support_agent import SupportAgent
    from app.logging_utils import LOGGER_NAME
    from app.schemas import KBSearchResultItem, SupportAgentInput, TriageResult
    from app.tools.bank_kb_tool import BankKnowledgeBaseTool


class StubClassifier:
    def __init__(self, response: object):
        self.response = response

    def invoke(self, input: dict[str, object], config: object | None = None, **kwargs: object) -> object:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class SupportAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.triage_result = TriageResult(
            is_bank_related=True,
            query_type="billing",
            needs_human=False,
            confidence=0.9,
            reason="Billing question.",
        )

    def test_stage_one_requests_kb_tool(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "use_tool",
                    "response_message": None,
                    "tool_name": "search_bank_kb",
                    "tool_query": "monthly maintenance fee explanation",
                    "needs_human": False,
                    "confidence": 0.84,
                    "reason": "Needs billing policy guidance.",
                }
            )
        )

        result = agent.handle(
            SupportAgentInput(user_query="Why was I charged a monthly maintenance fee?", triage_result=self.triage_result)
        )

        self.assertEqual(result.action, "use_tool")
        self.assertEqual(result.tool_name, "search_bank_kb")
        self.assertIsNone(result.response_message)

    def test_stage_one_can_respond_directly(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "respond",
                    "response_message": "You can usually check fee details in your statement and account terms.",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                    "confidence": 0.78,
                    "reason": "The question can be answered safely.",
                }
            )
        )

        result = agent.handle(
            SupportAgentInput(user_query="Where can I see my account fees?", triage_result=self.triage_result)
        )

        self.assertEqual(result.action, "respond")
        self.assertFalse(result.needs_human)

    def test_stage_one_can_escalate_manual_request(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "escalate",
                    "response_message": "This request requires review by a human support specialist.",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": True,
                    "confidence": 0.93,
                    "reason": "Transaction reversals require human review.",
                }
            )
        )

        result = agent.handle(
            SupportAgentInput(user_query="Please reverse this transaction", triage_result=self.triage_result)
        )

        self.assertEqual(result.action, "escalate")
        self.assertTrue(result.needs_human)

    def test_stage_two_uses_kb_results_for_final_response(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "respond",
                    "response_message": "Monthly maintenance fees may apply if the waiver conditions were not met.",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                    "confidence": 0.9,
                    "reason": "The issue was resolved using billing guidance.",
                }
            )
        )
        tool_result = [
            KBSearchResultItem(
                id="billing_1",
                title="Monthly Maintenance Fees",
                category="billing",
                content="Accounts may incur a monthly maintenance fee if they do not meet minimum balance or activity requirements.",
                confidence=0.92,
                source="billing.md",
            )
        ]

        result = agent.handle(
            SupportAgentInput(
                user_query="Why was I charged a monthly maintenance fee?",
                triage_result=self.triage_result,
                tool_result=tool_result,
            )
        )

        self.assertEqual(result.action, "respond")
        response_message = result.response_message
        self.assertIsNotNone(response_message)
        assert response_message is not None
        self.assertIn("waiver conditions", response_message)

    def test_stage_two_escalates_when_kb_is_weak(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "escalate",
                    "response_message": "This request requires review by a human support specialist.",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": True,
                    "confidence": 0.91,
                    "reason": "The available guidance is insufficient.",
                }
            )
        )

        result = agent.handle(
            SupportAgentInput(user_query="Please cancel this payment", triage_result=self.triage_result, tool_result=[])
        )

        self.assertEqual(result.action, "escalate")

    def test_malformed_output_falls_back_to_escalation(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "respond",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                    "confidence": 0.2,
                    "reason": "Missing message.",
                }
            )
        )

        result = agent.handle(
            SupportAgentInput(user_query="Help me understand this fee.", triage_result=self.triage_result)
        )

        self.assertEqual(result.action, "escalate")
        self.assertTrue(result.needs_human)

    def test_logs_support_phase_on_success(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "respond",
                    "response_message": "You can usually check fee details in your statement and account terms.",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                    "confidence": 0.78,
                    "reason": "The question can be answered safely.",
                }
            )
        )

        with self.assertLogs(f"{LOGGER_NAME}.support", level=logging.INFO) as captured:
            agent.handle(
                SupportAgentInput(user_query="Where can I see my account fees?", triage_result=self.triage_result)
            )

        output = "\n".join(captured.output)
        self.assertIn("PHASE 2 | Support Agent - Decision", output)
        self.assertIn("Tool results present: False", output)
        self.assertIn("Support result", output)

    def test_logs_support_fallback(self) -> None:
        agent = SupportAgent(
            classifier=StubClassifier(
                {
                    "action": "respond",
                    "tool_name": None,
                    "tool_query": None,
                    "needs_human": False,
                    "confidence": 0.2,
                    "reason": "Missing message.",
                }
            )
        )

        with self.assertLogs(f"{LOGGER_NAME}.support", level=logging.INFO) as captured:
            agent.handle(
                SupportAgentInput(user_query="Help me understand this fee.", triage_result=self.triage_result)
            )

        output = "\n".join(captured.output)
        self.assertIn("Support fallback triggered", output)
        self.assertIn("Support fallback result", output)


class BankKnowledgeBaseToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = BankKnowledgeBaseTool()

    def test_billing_query_prefers_billing_markdown(self) -> None:
        results = self.tool.search_bank_kb("monthly maintenance fee explanation", "billing")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "billing.md")
        self.assertIn("maintenance", results[0].title.lower())

    def test_technical_query_prefers_technical_markdown(self) -> None:
        results = self.tool.search_bank_kb("cannot log into mobile app", "technical")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "technical.md")
        self.assertIn("login", results[0].title.lower())

    def test_general_query_prefers_general_markdown(self) -> None:
        results = self.tool.search_bank_kb("customer support hours", "general")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "general.md")
        self.assertIn("support", results[0].title.lower())

    def test_faq_can_appear_as_supplementary_result(self) -> None:
        results = self.tool.search_bank_kb("monthly maintenance fee explanation", "billing")

        self.assertTrue(any(item.source == "faq.json" for item in results))

    def test_unmatched_query_does_not_crash(self) -> None:
        results = self.tool.search_bank_kb("galactic umbrella harmonics", "general")

        self.assertIsInstance(results, list)

    def test_logs_kb_phase(self) -> None:
        with self.assertLogs(f"{LOGGER_NAME}.kb", level=logging.INFO) as captured:
            self.tool.search_bank_kb("customer support hours", "general")

        output = "\n".join(captured.output)
        self.assertIn("PHASE 3 | Bank KB Tool", output)
        self.assertIn("Primary source: general.md", output)
        self.assertIn("KB tool results", output)


if __name__ == "__main__":
    unittest.main()
