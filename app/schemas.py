"""Shared request and response models for the bank support app."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


QueryType = Literal["technical", "billing", "general"]
NextAgent = Literal["support_agent", "escalation_agent"]
SupportAction = Literal["respond", "use_tool", "escalate"]
EscalationTrigger = Literal["triage", "support"]
EscalationPriority = Literal["medium", "high"]
EscalationDepartment = Literal["billing_support", "technical_support", "general_support", "fraud_support"]


class QueryRequest(BaseModel):
    """Incoming user query payload."""

    query: str = Field(min_length=1, description="The raw end-user support question.")


class TriageResult(BaseModel):
    """Structured triage output returned by the first agent."""

    model_config = ConfigDict(extra="forbid")

    is_bank_related: bool
    query_type: QueryType | None
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_non_bank_state(self) -> "TriageResult":
        if not self.is_bank_related:
            self.query_type = None
            self.needs_human = False
        return self


class ConversationContext(BaseModel):
    """Optional prior conversation context for future support iterations."""

    previous_messages: list[str] = Field(default_factory=list)


class KBSearchResultItem(BaseModel):
    """Normalized knowledge-base retrieval result."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    category: QueryType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class SupportAgentInput(BaseModel):
    """Input payload for the support agent."""

    user_query: str = Field(min_length=1)
    triage_result: TriageResult
    conversation_context: ConversationContext = Field(default_factory=ConversationContext)
    tool_result: list[KBSearchResultItem] | None = None


class SupportResult(BaseModel):
    """Structured support-agent output."""

    model_config = ConfigDict(extra="forbid")

    action: SupportAction
    response_message: str | None
    tool_name: str | None
    tool_query: str | None
    needs_human: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class EscalationContext(BaseModel):
    """Context describing why escalation was triggered."""

    triggered_by: EscalationTrigger
    escalation_reason: str = Field(min_length=1)


class EscalationAgentInput(BaseModel):
    """Input payload for the escalation agent."""

    user_query: str = Field(min_length=1)
    triage_result: TriageResult
    support_result: SupportResult | None = None
    escalation_context: EscalationContext


class EscalationResult(BaseModel):
    """Structured escalation-agent output."""

    model_config = ConfigDict(extra="forbid")

    customer_message: str = Field(min_length=1)
    handoff_summary: str = Field(min_length=1)
    priority: EscalationPriority
    department: EscalationDepartment
    reason: str = Field(min_length=1)


class RouteResponse(BaseModel):
    """Orchestrator response describing triage plus the next route."""

    triage: TriageResult
    support: SupportResult | None = None
    tool_result: list[KBSearchResultItem] | None = None
    escalation: EscalationResult | None = None
    next_agent: NextAgent | None
