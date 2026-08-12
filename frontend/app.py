"""Streamlit frontend for the AI Shopping Decision Assistant."""
from __future__ import annotations

import time
import uuid

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"
PAGE_TITLE = "AI Shopping Assistant"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_categories() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/api/categories", timeout=10)
        resp.raise_for_status()
        return [""] + resp.json().get("categories", [])
    except Exception:  # noqa: BLE001
        return [""]


def api_search(
    query: str,
    category: str,
    price_min: float,
    price_max: float,
    min_rating: float,
) -> dict:
    payload = {
        "query": query,
        "category": category,
        "price_min": price_min,
        "price_max": price_max,
        "min_rating": min_rating,
        "session_id": st.session_state.session_id,
    }
    resp = httpx.post(f"{BACKEND_URL}/api/search", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def api_chat(message: str, product_context: list[dict]) -> str:
    payload = {
        "message": message,
        "session_id": st.session_state.session_id,
        "context": product_context,
    }
    resp = httpx.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    st.session_state.session_id = data.get("session_id", st.session_state.session_id)
    return data.get("reply", "")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_star_rating(rating: float) -> str:
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def render_product_card(product: dict, rank: int) -> None:
    rank_emoji = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
    emoji = rank_emoji[rank - 1] if rank <= len(rank_emoji) else "🏅"

    ai_score = product.get("ai_score")
    reason = product.get("reason", "")
    tradeoffs = product.get("tradeoffs", "")
    pros = product.get("advantages") or product.get("pros_llm", [])
    cons = product.get("disadvantages") or product.get("cons_llm", [])

    with st.container():
        col_img, col_info = st.columns([1, 4])

        with col_img:
            if product.get("thumbnail"):
                st.image(product["thumbnail"], width=120)

        with col_info:
            title_line = f"{emoji} **#{rank} {product.get('title', 'Unknown')}**"
            if ai_score is not None:
                title_line += f"  &nbsp; AI Score: `{ai_score:.2f}`"
            st.markdown(title_line)

            price = product.get("price", 0)
            discount = product.get("discountPercentage", 0)
            discounted_price = price * (1 - discount / 100)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", f"${discounted_price:.2f}", f"-{discount:.0f}%" if discount else None)
            c2.metric("Rating", f"{product.get('rating', 0):.1f}/5", render_star_rating(product.get("rating", 0)))
            c3.metric("Stock", str(product.get("stock", 0)), product.get("availabilityStatus", ""))
            c4.metric("Brand", product.get("brand", "N/A"))

            if reason:
                st.info(f"💡 **Reason:** {reason}")
            if tradeoffs:
                st.warning(f"⚖️ **Tradeoffs:** {tradeoffs}")

            if pros or cons:
                pc1, pc2 = st.columns(2)
                if pros:
                    with pc1:
                        st.markdown("**✅ Advantages**")
                        for p in pros:
                            st.markdown(f"- {p}")
                if cons:
                    with pc2:
                        st.markdown("**❌ Disadvantages**")
                        for c in cons:
                            st.markdown(f"- {c}")

            details_cols = st.columns(2)
            details_cols[0].caption(f"🛡️ {product.get('warrantyInformation', 'N/A')}")
            details_cols[1].caption(f"↩️ {product.get('returnPolicy', 'N/A')}")

        st.divider()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🛍️ AI Shopping Decision Assistant")
st.caption("Compare products intelligently — powered by Azure OpenAI + LangChain")

# Sidebar — filters
with st.sidebar:
    st.header("🔍 Search & Filters")

    categories = fetch_categories()
    selected_category = st.selectbox(
        "Category",
        categories,
        format_func=lambda x: x.replace("-", " ").title() if x else "All Categories",
    )

    query = st.text_input("Search query", placeholder="e.g. cushioned running shoes")

    st.subheader("Price Range ($)")
    price_col1, price_col2 = st.columns(2)
    price_min = price_col1.number_input("Min", min_value=0.0, max_value=10000.0, value=0.0, step=10.0)
    price_max = price_col2.number_input("Max", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)

    min_rating = st.slider("Minimum Rating ⭐", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

    search_clicked = st.button("🔎 Search & Compare", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

    if st.button("🔄 Refresh Catalog"):
        try:
            httpx.post(f"{BACKEND_URL}/api/catalog/refresh", timeout=30)
            fetch_categories.clear()
            st.success("Catalog refreshed!")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refresh failed: {exc}")

# Main content area
tab_results, tab_chat = st.tabs(["📦 Results", "💬 Chat"])

# ---------------------------------------------------------------------------
# Results tab
# ---------------------------------------------------------------------------

with tab_results:
    if search_clicked:
        if price_min > price_max:
            st.error("Min price cannot exceed max price.")
        else:
            with st.spinner("Searching and comparing products with AI…"):
                try:
                    t0 = time.time()
                    data = api_search(
                        query=query,
                        category=selected_category,
                        price_min=price_min,
                        price_max=price_max,
                        min_rating=min_rating,
                    )
                    elapsed = time.time() - t0

                    results = data.get("results", [])
                    summary = data.get("summary", "")
                    errors = data.get("errors")

                    st.session_state.last_results = results
                    st.session_state.last_summary = summary

                    if errors:
                        st.warning(f"⚠️ {errors}")

                    st.success(f"✅ {summary}  ·  _{elapsed:.1f}s_")

                    if not results:
                        st.info("No products found. Try adjusting your filters.")
                    else:
                        st.subheader(f"Top {len(results)} Products")
                        for i, product in enumerate(results):
                            render_product_card(product, rank=i + 1)

                except httpx.HTTPStatusError as exc:
                    detail = exc.response.json().get("detail", str(exc))
                    st.error(f"API error: {detail}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not reach the backend: {exc}")
                    st.info("Make sure the backend server is running: `uvicorn backend.main:app --reload`")

    elif st.session_state.last_results:
        st.info(f"Showing last search: _{st.session_state.last_summary}_")
        for i, product in enumerate(st.session_state.last_results):
            render_product_card(product, rank=i + 1)
    else:
        st.info("👈 Set your filters and click **Search & Compare** to get started.")

        # Quick-start examples
        st.subheader("Try these searches:")
        examples = [
            ("📱 Smartphones under $500", "smartphones", 0, 500, 4.0),
            ("👟 Sports shoes $30–$100", "mens-shoes", 30, 100, 3.5),
            ("💻 Laptops with high rating", "laptops", 0, 2000, 4.5),
            ("🎧 All electronics", "", 0, 1000, 4.0),
        ]
        cols = st.columns(2)
        for i, (label, cat, pmin, pmax, mrat) in enumerate(examples):
            if cols[i % 2].button(label, use_container_width=True):
                with st.spinner("Searching…"):
                    try:
                        data = api_search(label, cat, pmin, pmax, mrat)
                        st.session_state.last_results = data.get("results", [])
                        st.session_state.last_summary = data.get("summary", "")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

with tab_chat:
    st.subheader("💬 Ask a Follow-up Question")
    st.caption("Ask anything about the products in the current search results.")

    # Display history
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    # Input
    if user_msg := st.chat_input("e.g. Which one is best for daily commuting?"):
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        # Build minimal product context from last results
        product_context = [
            {
                "title": p.get("title"),
                "price": p.get("price"),
                "rating": p.get("rating"),
                "stock": p.get("stock"),
                "returnPolicy": p.get("returnPolicy"),
                "availabilityStatus": p.get("availabilityStatus"),
            }
            for p in st.session_state.last_results[:5]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = api_chat(user_msg, product_context)
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as exc:  # noqa: BLE001
                    err = f"Chat error: {exc}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
