# Bank Support Multi-Agent

A local multi-agent banking support API built with FastAPI, LangChain, and Ollama.

The project routes a user query through three specialized agents:

- `Triage Agent`: decides whether the query is bank-related, classifies it, and detects immediate escalation
- `Support Agent`: decides whether it can answer directly, needs the bank knowledge base, or should escalate
- `Escalation Agent`: prepares the final human handoff with a customer-facing message and an internal summary

## Current Flow

`POST /route` runs the full orchestration:

1. Triage classifies the query
2. If triage says human review is needed, escalation runs immediately
3. Otherwise support decides whether to answer or use `search_bank_kb`
4. If retrieval is needed, the KB tool runs with query-type-aware routing
5. Support produces the final grounded response or escalates
6. If support escalates, escalation prepares the final handoff

`POST /triage` is also available for debugging just the first phase.

## Features

- Local LLM integration through Ollama
- Structured outputs validated with Pydantic
- Query-type-aware KB search:
  - `billing.md`
  - `technical.md`
  - `general.md`
  - `faq.json` as supplementary retrieval context
- Terminal phase logging for:
  - triage
  - support decision
  - KB retrieval
  - support final response
  - escalation
- Unit tests for triage, support, KB search, orchestration, and escalation

## Project Structure

```text
bank_support_multi_agent/
├── app/
│   ├── agents/
│   │   ├── triage_agent.py
│   │   ├── support_agent.py
│   │   └── escalation_agent.py
│   ├── knowledge_base/
│   │   ├── billing.md
│   │   ├── technical.md
│   │   ├── general.md
│   │   └── faq.json
│   ├── prompts/
│   │   ├── triage_prompt.txt
│   │   ├── support_prompt.txt
│   │   └── escalation_prompt.txt
│   ├── tools/
│   │   └── bank_kb_tool.py
│   ├── config.py
│   ├── logging_utils.py
│   ├── main.py
│   ├── orchestrator.py
│   └── schemas.py
├── tests/
│   ├── test_triage.py
│   ├── test_support.py
│   ├── test_bank_kb_tool.py
│   ├── test_orchestrator.py
│   └── test_escalation.py
└── requirements.txt
```

## Requirements

- Python 3.10+
- Ollama installed on Windows
- A local Ollama model available, currently configured for `llama3.1`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the Ollama model if needed:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" pull llama3.1
```

## Configuration

Environment variables used by the app:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
API_HOST=127.0.0.1
API_PORT=8000
```

You can place them in `.env`, but the current config also has defaults if they are not set.

## Run the API

From inside `bank_support_multi_agent`:

```powershell
python -m uvicorn app.main:app --reload
```

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

## Example Requests

Triage only:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/triage" -ContentType "application/json" -Body '{"query":"Why was I charged a monthly fee?"}'
```

Full orchestration:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/route" -ContentType "application/json" -Body '{"query":"I cant log into the app"}'
```

Billing example:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/route" -ContentType "application/json" -Body '{"query":"Why was I charged a monthly maintenance fee?"}'
```

Fraud example:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/route" -ContentType "application/json" -Body '{"query":"I see transactions I did not make"}'
```

## API Response Shape

`POST /route` returns a response like:

```json
{
  "triage": {
    "is_bank_related": true,
    "query_type": "billing",
    "needs_human": false,
    "confidence": 0.9,
    "reason": "The user is asking about a bank fee."
  },
  "support": {
    "action": "use_tool",
    "response_message": null,
    "tool_name": "search_bank_kb",
    "tool_query": "monthly maintenance fee explanation",
    "needs_human": false,
    "confidence": 0.84,
    "reason": "Needs billing policy guidance."
  },
  "tool_result": [],
  "escalation": null,
  "next_agent": "support_agent"
}
```

Depending on the route, `support`, `tool_result`, or `escalation` can be `null`.

## Terminal Logging

The app logs each phase in the terminal where uvicorn is running:

- `PHASE 1 | Triage Agent`
- `PHASE 2 | Support Agent - Decision`
- `PHASE 3 | Bank KB Tool`
- `PHASE 4 | Support Agent - Final Response`
- `PHASE 5 | Escalation Agent`

Logs include:

- phase banners
- model connection/waiting notes
- route decisions
- normalized JSON outputs
- fallback/error information when parsing or validation fails

## Run Tests

From inside `bank_support_multi_agent`:

```powershell
python -m unittest tests.test_triage
python -m unittest tests.test_support
python -m unittest tests.test_bank_kb_tool
python -m unittest tests.test_orchestrator
python -m unittest tests.test_escalation
```

Or run the whole suite:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- The project currently assumes Ollama is available locally.
- If Ollama is installed but not on your `PATH`, use the full executable path:

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

- If a model response is malformed, the agents fall back safely to escalation-oriented behavior instead of guessing.
