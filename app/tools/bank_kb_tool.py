"""Knowledge-base search tool for bank support routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ..logging_utils import get_logger, log_json, log_phase
from ..schemas import KBSearchResultItem, QueryType


@dataclass(frozen=True)
class _SourceDocument:
    id: str
    title: str
    category: QueryType
    content: str
    source: str
    base_confidence: float
    source_weight: float


class BankKnowledgeBaseTool:
    """Searches the bank KB using query-type-aware source selection."""

    def __init__(self, kb_dir: Path | None = None) -> None:
        self._kb_dir = kb_dir or Path(__file__).resolve().parents[1] / "knowledge_base"
        self._logger = get_logger("kb")

    def search_bank_kb(self, query: str, query_type: QueryType | None, limit: int = 5) -> list[KBSearchResultItem]:
        """Return ranked KB matches for the user query."""

        log_phase(self._logger, "PHASE 3 | Bank KB Tool")
        self._logger.info("Tool selected: search_bank_kb")
        self._logger.info("Primary source: %s", f"{query_type}.md" if query_type is not None else "faq.json")
        self._logger.info("FAQ supplementary source enabled: %s", query_type is not None)
        normalized_query = query.strip()
        if not normalized_query:
            self._logger.info("Empty tool query received")
            return []

        documents = self._load_documents(query_type)
        if not documents:
            return []

        ranked: list[tuple[float, _SourceDocument]] = []
        for document in documents:
            score = self._score_document(normalized_query, query_type, document)
            if score <= 0:
                continue
            ranked.append((score, document))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results = [
            KBSearchResultItem(
                id=document.id,
                title=document.title,
                category=document.category,
                content=document.content,
                confidence=min(1.0, score),
                source=document.source,
            )
            for score, document in ranked[:limit]
        ]
        self._logger.info("Top matches returned: %s", len(results))
        log_json(self._logger, "KB tool results", [item.model_dump() for item in results])
        return results

    def _load_documents(self, query_type: QueryType | None) -> list[_SourceDocument]:
        if query_type is None:
            return self._load_faq_documents(None)
        return self._load_markdown_documents(query_type) + self._load_faq_documents(query_type)

    def _load_markdown_documents(self, query_type: QueryType) -> list[_SourceDocument]:
        path = self._kb_dir / f"{query_type}.md"
        if not path.exists():
            return []

        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []

        pattern = re.compile(r"^##\s+(.+?)\n(.*?)(?=^##\s+|\Z)", re.MULTILINE | re.DOTALL)
        matches = list(pattern.finditer(raw_text))
        documents: list[_SourceDocument] = []
        for index, match in enumerate(matches, start=1):
            title = re.sub(r"^\d+\.\s*", "", match.group(1).strip())
            content = "\n".join(line.strip() for line in match.group(2).splitlines() if line.strip()) or title
            documents.append(
                _SourceDocument(
                    id=f"{query_type}_{index}",
                    title=title,
                    category=query_type,
                    content=content,
                    source=f"{query_type}.md",
                    base_confidence=0.85,
                    source_weight=0.35,
                )
            )
        if documents:
            return documents

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return []
        return [
            _SourceDocument(
                id=f"{query_type}_1",
                title=re.sub(r"^\d+\.\s*", "", lines[0].lstrip("#").strip()),
                category=query_type,
                content="\n".join(lines[1:]).strip() or lines[0],
                source=f"{query_type}.md",
                base_confidence=0.85,
                source_weight=0.35,
            )
        ]

    def _load_faq_documents(self, query_type: QueryType | None) -> list[_SourceDocument]:
        path = self._kb_dir / "faq.json"
        if not path.exists():
            return []

        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return []

        payload = json.loads(raw_text)
        documents: list[_SourceDocument] = []
        for item in payload:
            category = cast(QueryType, item["category"])
            if query_type is None:
                weight = 0.2
            elif category == query_type:
                weight = 0.18
            else:
                weight = 0.05
            documents.append(
                _SourceDocument(
                    id=item["id"],
                    title=item["title"],
                    category=category,
                    content="\n".join(
                        [
                            item["question"],
                            item["answer"],
                            item.get("applicability", ""),
                        ]
                    ).strip(),
                    source="faq.json",
                    base_confidence=float(item.get("confidence", 0.75)),
                    source_weight=weight,
                )
            )
        return documents

    def _score_document(self, query: str, query_type: QueryType | None, document: _SourceDocument) -> float:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return 0.0

        title_text = self._normalize(document.title)
        content_text = self._normalize(document.content)
        overlap = len(query_tokens & self._tokenize(document.title + " " + document.content))
        token_score = overlap / len(query_tokens)

        normalized_query = self._normalize(query)
        phrase_score = 0.0
        if normalized_query in title_text:
            phrase_score += 0.3
        elif normalized_query in content_text:
            phrase_score += 0.2

        if overlap == 0 and phrase_score == 0.0:
            return 0.0

        category_score = 0.12 if query_type is not None and document.category == query_type else 0.0
        total = (document.base_confidence * 0.35) + token_score + phrase_score + document.source_weight + category_score
        return round(min(1.0, total), 4)

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", self._normalize(text)) if len(token) > 1}

    def _normalize(self, text: str) -> str:
        return text.lower().replace("â€“", "-").replace("â†’", "->").replace("â€™", "'")
