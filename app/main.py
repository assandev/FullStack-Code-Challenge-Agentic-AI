"""FastAPI entrypoint for the bank support multi-agent demo."""

from __future__ import annotations

from fastapi import FastAPI

from .agents.escalation_agent import EscalationAgent
from .agents.support_agent import SupportAgent
from .agents.triage_agent import TriageAgent
from .config import get_settings
from .logging_utils import get_logger
from .orchestrator import SupportOrchestrator
from .schemas import QueryRequest, RouteResponse, TriageResult
from .tools.bank_kb_tool import BankKnowledgeBaseTool

settings = get_settings()
app = FastAPI(title="Bank Support Multi-Agent")
logger = get_logger("api")

triage_agent = TriageAgent()
support_agent = SupportAgent()
escalation_agent = EscalationAgent()
kb_tool = BankKnowledgeBaseTool()
orchestrator = SupportOrchestrator(
    triage_agent=triage_agent,
    support_agent=support_agent,
    kb_tool=kb_tool,
    escalation_agent=escalation_agent,
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Basic health endpoint for local development."""

    return {"status": "ok", "model": settings.ollama_model}


@app.post("/triage", response_model=TriageResult)
def triage(request: QueryRequest) -> TriageResult:
    """Return the raw triage result for a query."""

    logger.info("API /triage request started")
    result = triage_agent.classify(request.query)
    logger.info("API /triage request completed")
    return result


@app.post("/route", response_model=RouteResponse)
def route(request: QueryRequest) -> RouteResponse:
    """Return triage plus the next selected agent route."""

    logger.info("API /route request started")
    result = orchestrator.route(request.query)
    logger.info("API /route request completed")
    return result
