"""Tests for the triage agent and response normalization."""

from __future__ import annotations

import logging
import unittest

try:
    from bank_support_multi_agent.app.agents.triage_agent import TriageAgent
    from bank_support_multi_agent.app.logging_utils import LOGGER_NAME
    from bank_support_multi_agent.app.schemas import TriageResult
except ModuleNotFoundError:
    from app.agents.triage_agent import TriageAgent
    from app.logging_utils import LOGGER_NAME
    from app.schemas import TriageResult


class StubClassifier:
    """Simple stub that mimics a LangChain runnable."""

    def __init__(self, response: object):
        self.response = response

    def invoke(self, input: dict[str, object], config: object | None = None, **kwargs: object) -> object:
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class TriageAgentTests(unittest.TestCase):
    def test_billing_query_classification(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "billing",
                    "needs_human": False,
                    "confidence": 0.87,
                    "reason": "The user is asking about an unexpected monthly fee.",
                }
            )
        )

        result = agent.classify("Why do I have a monthly fee on my account?")

        self.assertEqual(result.query_type, "billing")
        self.assertTrue(result.is_bank_related)
        self.assertFalse(result.needs_human)

    def test_technical_query_classification(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "technical",
                    "needs_human": False,
                    "confidence": 0.91,
                    "reason": "The user cannot sign in to the banking app.",
                }
            )
        )

        result = agent.classify("I can't log into the app")

        self.assertEqual(result.query_type, "technical")

    def test_general_query_classification(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "general",
                    "needs_human": False,
                    "confidence": 0.83,
                    "reason": "The user is asking about support hours.",
                }
            )
        )

        result = agent.classify("What time does customer support close?")

        self.assertEqual(result.query_type, "general")

    def test_non_bank_query_is_forced_to_null_route(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": False,
                    "query_type": "general",
                    "needs_human": True,
                    "confidence": 0.99,
                    "reason": "The query is about the weather, not banking.",
                }
            )
        )

        result = agent.classify("What's the weather today?")

        self.assertFalse(result.is_bank_related)
        self.assertIsNone(result.query_type)
        self.assertFalse(result.needs_human)

    def test_fraud_query_requests_human_review(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "billing",
                    "needs_human": True,
                    "confidence": 0.96,
                    "reason": "The user reports suspicious transactions.",
                }
            )
        )

        result = agent.classify("I see transactions I didn't make")

        self.assertTrue(result.needs_human)
        self.assertEqual(result.query_type, "billing")

    def test_malformed_llm_output_uses_safe_fallback(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "billing",
                    "needs_human": False,
                    "confidence": 0.3,
                }
            )
        )

        result = agent.classify("Please help")

        self.assertTrue(result.is_bank_related)
        self.assertTrue(result.needs_human)
        self.assertIsNone(result.query_type)
        self.assertEqual(result.confidence, 0.0)

    def test_confidence_is_clamped(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "general",
                    "needs_human": False,
                    "confidence": 1.4,
                    "reason": "The user asks a general bank question.",
                }
            )
        )

        result = agent.classify("How do I update my address?")

        self.assertEqual(result.confidence, 1.0)

    def test_logs_phase_and_json_on_success(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "billing",
                    "needs_human": False,
                    "confidence": 0.87,
                    "reason": "The user is asking about an unexpected monthly fee.",
                }
            )
        )

        with self.assertLogs(f"{LOGGER_NAME}.triage", level=logging.INFO) as captured:
            agent.classify("Why do I have a monthly fee on my account?")

        output = "\n".join(captured.output)
        self.assertIn("PHASE 1 | Triage Agent", output)
        self.assertIn("Connected to", output)
        self.assertIn("Waiting for Llama", output)
        self.assertIn("Json returned", output)
        self.assertIn("Triage result", output)

    def test_logs_warning_on_fallback(self) -> None:
        agent = TriageAgent(
            classifier=StubClassifier(
                {
                    "is_bank_related": True,
                    "query_type": "billing",
                    "needs_human": False,
                    "confidence": 0.3,
                }
            )
        )

        with self.assertLogs(f"{LOGGER_NAME}.triage", level=logging.INFO) as captured:
            agent.classify("Please help")

        output = "\n".join(captured.output)
        self.assertIn("Triage fallback triggered", output)
        self.assertIn("Triage fallback result", output)


if __name__ == "__main__":
    unittest.main()
