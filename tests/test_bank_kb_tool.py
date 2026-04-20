"""Focused tests for query-type-aware KB search."""

from __future__ import annotations

import unittest

try:
    from bank_support_multi_agent.app.tools.bank_kb_tool import BankKnowledgeBaseTool
except ModuleNotFoundError:
    from app.tools.bank_kb_tool import BankKnowledgeBaseTool


class BankKnowledgeBaseToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = BankKnowledgeBaseTool()

    def test_billing_query_ranks_primary_markdown_before_faq(self) -> None:
        results = self.tool.search_bank_kb("refund processing time", "billing")

        self.assertTrue(results)
        self.assertEqual(results[0].source, "billing.md")

    def test_null_query_type_uses_faq_only(self) -> None:
        results = self.tool.search_bank_kb("customer support hours", None)

        self.assertTrue(results)
        self.assertTrue(all(item.source == "faq.json" for item in results))


if __name__ == "__main__":
    unittest.main()
