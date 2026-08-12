"""Conversational chat module — maintains session context and handles follow-up questions.

Uses LangChain v1 ConversationChain pattern with Azure OpenAI.
Session history is stored in-memory (demo); swap for Redis in production.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

from backend.config import settings
from backend.security import sanitize_input, session_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session store (in-memory)
# ---------------------------------------------------------------------------

_sessions: dict[str, InMemoryChatMessageHistory] = {}


def _get_or_create_session(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _sessions:
        history = InMemoryChatMessageHistory()
        history.add_message(
            SystemMessage(content=(
                "You are a helpful AI shopping assistant. "
                "Help users compare products, understand tradeoffs, and make confident purchase decisions. "
                "Be concise, specific, and always base your answers on the product context provided. "
                "If the user asks about products outside the provided context, say so clearly."
            ))
        )
        _sessions[session_id] = history
    return _sessions[session_id]


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _get_or_create_session(session_id)
    return session_id


def get_session_messages(session_id: str) -> list[dict[str, str]]:
    """Return session history as a list of role/content dicts."""
    if session_id not in _sessions:
        return []
    history = _sessions[session_id]
    result = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


# ---------------------------------------------------------------------------
# LLM chat
# ---------------------------------------------------------------------------

def _build_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        azure_deployment=settings.azure_openai_deployment_name,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.5,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    )


_llm: AzureChatOpenAI | None = None


def _get_llm() -> AzureChatOpenAI:
    global _llm  # noqa: PLW0603
    if _llm is None:
        _llm = _build_llm()
    return _llm


def chat(
    message: str,
    session_id: str,
    product_context: list[dict[str, Any]] | None = None,
) -> tuple[str, str]:
    """Process a user chat message and return (reply, session_id).

    If product_context is provided (list of product dicts), it is injected
    into the message as context for the LLM.
    """
    cleaned_message = sanitize_input(message, 200)
    if not cleaned_message:
        return "Message cannot be empty.", session_id or create_session()

    if not session_id:
        session_id = create_session()

    session_store.update_session(
        session_id,
        {
            "current_query": cleaned_message,
            "recent_context": [p.get("id") for p in (product_context or [])[:10]],
        },
    )

    history = _get_or_create_session(session_id)

    # Inject product context if available
    augmented_message = cleaned_message
    if product_context:
        context_lines = []
        for p in product_context[:5]:
            context_lines.append(
                f"- {p.get('title')} (${p.get('price')}, rating: {p.get('rating')}, "
                f"stock: {p.get('stock')}, return: {p.get('returnPolicy', 'N/A')})"
            )
        context_text = "\n".join(context_lines)
        augmented_message = (
            f"[Current product context for this session:]\n{context_text}\n\n"
            f"[User question:] {cleaned_message}"
        )

    history.add_user_message(augmented_message)

    try:
        llm = _get_llm()
        reply = llm.invoke(history.messages).content
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat LLM error: %s", exc)
        reply = (
            "I'm having trouble connecting to the AI service right now. "
            "Please try again in a moment."
        )

    history.add_ai_message(reply)
    return reply, session_id
