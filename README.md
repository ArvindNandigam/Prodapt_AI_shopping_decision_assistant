# AI Shopping Decision Assistant

An AI-powered product comparison assistant that helps users filter, rank, and understand products based on their needs, budget, and preferences — with explainable rationale on every recommendation.

Built with **FastAPI + Streamlit + LangChain 1.3.15 + Azure OpenAI**, using the [dummyjson.com](https://dummyjson.com/docs/products) product catalog (194 SKUs, 24 categories).

# PPT link : https://gamma.app/docs/AI-Shopping-Decision-Assistant-a993i4hlri62t37
---

## Features

- **Smart search** — filter by category, price range, and minimum rating
- **AI comparison** — LangChain-powered ranking with pros, cons, tradeoffs, and a score per product
- **Conversational follow-ups** — ask follow-up questions with full session context
- **Graceful fallback** — if the LLM is unavailable, raw results sorted by rating are still shown
- **Catalog caching** — dummyjson is fetched once at startup and cached locally; a snapshot on disk acts as fallback if the API is unreachable
- **PII protection** — `reviewerName` and `reviewerEmail` are stripped at cache time, before any data reaches a prompt or log

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

The backend starts at `http://localhost:8000`. On startup it fetches the full dummyjson catalog and caches it in memory.

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

37 tests covering search logic, catalog PII stripping, comparator fallback/parsing, and all API endpoints.

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
