# AI Shopping Decision Assistant

An AI-powered product comparison assistant that helps users filter, rank, and understand products based on their needs, budget, and preferences — with explainable rationale on every recommendation.

Built with **FastAPI + Streamlit + LangChain 1.3.15 + Azure OpenAI**, using the [dummyjson.com](https://dummyjson.com/docs/products) product catalog (194 SKUs, 24 categories).

# PPT link : https://gamma.app/docs/AI-Shopping-Decision-Assistant-2f0xoyki6wybf42
---

## Features

- **Smart search** — filter by category, price range, and minimum rating
- **AI comparison** — LangChain-powered ranking with pros, cons, tradeoffs, and a score per product
- **Conversational follow-ups** — ask follow-up questions with full session context
- **Graceful fallback** — if the LLM is unavailable, raw results sorted by rating are still shown
- **Catalog caching** — dummyjson is fetched once at startup and cached locally; a snapshot on disk acts as fallback if the API is unreachable
- **PII protection** — `reviewerName` and `reviewerEmail` are stripped at cache time, before any data reaches a prompt or log
- **Incognito sessions** — temporary in-memory session state with a 30-minute inactivity TTL; no long-term personal profile or permanent memory
- **Execution auditing** — SQLite-based trace log captures request lifecycle, tool/search steps, validation outcomes, and recommendation status without storing chain-of-thought or sensitive data
- **Input guardrails** — request validation for query, category, budget, minimum rating, and result limits
- **Prompt-injection guardrails** — user text is sanitized and sensitive or override-style phrases are removed before model interaction
- **Recommendation safety** — the app validates AI output against budget/rating/product constraints and falls back to valid filtered products when needed

---

## Security, Privacy & Reliability

The backend adds a lightweight trust boundary around the existing LangChain workflow so it remains hackathon-friendly while still protecting the demo from obvious misuse.

### Privacy / Incognito

- Temporary session context only: `session_id`, current query, temporary preferences, budget, minimum rating, recent context, and candidate product IDs
- No persistent user profile is created
- No names, emails, phones, addresses, or long-term personal preferences are stored
- Session data expires automatically after 30 minutes of inactivity

### Audit trail / accountability

- A local SQLite database stores execution metadata such as `trace_id`, `session_id`, timestamp, status, step, latency, product identifiers, validation result, and recommendation outcome
- The audit trail is intentionally non-PII and excludes API keys, passwords, chain-of-thought, and full conversation content
- This makes it possible to answer: what happened in this agent run, which tool/search steps ran, what succeeded or failed, and what recommendation was returned

### Input and prompt guardrails

- Query length is capped at 200 characters
- Category length is capped at 100 characters
- Rating must stay between 0 and 5
- Price values must remain non-negative
- `price_min` cannot exceed `price_max`
- Search limit is capped at 20
- Prompt-injection style phrases such as override/bypass/security prompt wording are sanitized before the model sees the content

### Recommendation integrity

- LLM output is not trusted blindly
- The app validates returned products against the retrieved candidate set, score bounds, ranking, and hard constraints
- If the model invents a product ID, generates invalid numeric values, or violates budget/rating rules, the backend rejects or corrects the result and falls back safely

### Reliability and fallback

- Search and AI operations are wrapped in a safe fallback path
- If the LLM fails or times out, the app still returns valid filtered products instead of fabricating a recommendation
- This preserves the user experience while keeping the system safe and explainable

---

## Database & Storage Model

### Runtime storage

- **In-memory session store**: used for temporary chat/search session data with TTL expiry
- **SQLite audit database**: local file at `backend/audit.db` for non-PII execution logs
- **Catalog cache**: in-memory list of product data loaded from dummyjson at startup, with a snapshot fallback on disk if the API is unavailable

### Data retention policy

- Session state is short-lived and expires automatically
- Audit events are stored for operational accountability, not for long-term personal memory
- No sensitive personal data is recorded or resolved into persistent profile state
- Product review names/emails are stripped before data enters prompts or logs

---

## Project Structure

```
.
├── backend/
│   ├── main.py          # FastAPI app, routes, lifespan
│   ├── catalog.py       # dummyjson fetch, in-memory cache, snapshot fallback
│   ├── search.py        # Structured filter + full-text search over the cache
│   ├── comparator.py    # LangChain LLM comparator skill (LCEL chain)
│   ├── chat.py          # Conversational chat with session history
│   ├── models.py        # Pydantic data models
│   ├── security.py      # Session TTL, audit logging, validation, sanitization, rate limiting
│   └── config.py        # Settings via pydantic-settings (.env)
├── frontend/
│   └── app.py           # Streamlit UI
├── tests/
│   ├── test_api.py      # FastAPI endpoint integration tests
│   ├── test_catalog.py  # Catalog loader + PII stripping unit tests
│   ├── test_comparator.py # Comparator fallback + output parsing tests
│   └── test_search.py   # Search engine unit tests
├── data/                # Auto-created; stores catalog_snapshot.json
├── .env                 # Azure OpenAI credentials (not committed)
├── .env.example         # Credential template
├── pyproject.toml       # uv-managed dependencies
├── run_backend.bat      # Start the FastAPI backend
└── run_frontend.bat     # Start the Streamlit frontend
```

---

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager
- Python 3.11+
- An Azure OpenAI resource with a deployment (e.g. `gpt-4o-mini`)

### 1. Clone and configure

```bash
git clone https://github.com/your-org/Prodapt_AI_shopping_decision_assistant.git
cd Prodapt_AI_shopping_decision_assistant
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Run the backend

```bash
# Windows
run_backend.bat

# Or directly
uv run uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend starts at `http://localhost:8000`. On startup it fetches the full dummyjson catalog and caches it in memory. It also initializes the temporary session store and the local SQLite execution audit DB for non-PII trace logging.

### 4. Run the frontend

Open a second terminal:

```bash
# Windows
run_frontend.bat

# Or directly
uv run streamlit run frontend/app.py --server.port 8501
```

Open `http://localhost:8501` in your browser.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

| Variable                       | Description                    | Default              |
| ------------------------------ | ------------------------------ | -------------------- |
| `AZURE_OPENAI_API_KEY`         | Azure OpenAI API key           | required             |
| `AZURE_OPENAI_ENDPOINT`        | Azure OpenAI endpoint URL      | required             |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Deployment name                | `gpt-4o-mini`        |
| `AZURE_OPENAI_API_VERSION`     | API version                    | `2024-12-01-preview` |
| `USE_LLM`                      | Enable/disable LLM comparison  | `true`               |
| `SEARCH_TOP_K`                 | Max candidates sent to the LLM | `10`                 |

---

## API Endpoints

| Method | Path                   | Description                    |
| ------ | ---------------------- | ------------------------------ |
| `GET`  | `/health`              | Liveness check + catalog stats |
| `GET`  | `/api/categories`      | List all product categories    |
| `POST` | `/api/search`          | Search + AI-rank products      |
| `POST` | `/api/chat`            | Conversational follow-up       |
| `POST` | `/api/catalog/refresh` | Re-sync catalog from dummyjson |

Interactive docs at `http://localhost:8000/docs`.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

42 tests covering search logic, catalog PII stripping, comparator fallback/parsing, security validation, and all API endpoints.

---

## Tech Stack

| Layer             | Technology                                |
| ----------------- | ----------------------------------------- |
| Frontend          | Streamlit 1.41                            |
| Backend           | FastAPI 0.115 + Uvicorn                   |
| LLM orchestration | LangChain 1.3.15 + langchain-openai 1.4.3 |
| LLM provider      | Azure OpenAI (GPT-4o-mini)                |
| Product data      | dummyjson.com REST API                    |
| Data validation   | Pydantic v2                               |
| Package manager   | uv                                        |

---

## Architecture

See [`arch.md`](arch.md) for the full architecture document covering data modeling, prompt engineering, security/PII policy, scalability strategy, and the component + sequence diagrams.

Current runtime trust boundaries:

- User input is sanitized before prompt injection prevention and validation
- Search + candidate retrieval happens in the deterministic layer
- The LLM comparator is only given filtered, trimmed candidate data
- Outputs are validated against budget, rating, product-ID, rank, and score constraints
- Audit events are logged to SQLite without storing secrets, chain-of-thought, or personal data
- If the LLM fails, the app falls back to a safe recommendation path using valid filtered products
