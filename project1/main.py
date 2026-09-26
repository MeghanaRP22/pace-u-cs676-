"""
CS676 Project 1 — Credibility-scored research chatbot.

A Streamlit chat app that answers questions using Claude, shows the sources it
used, and displays a credibility score beside each one.

Run it with:   streamlit run main.py

Changes for this submission: sources are scored in parallel and cached, each
chip shows the score, its 90% range and a source-type label, the "Why this
score?" expander lists the evidence, and the sidebar lets you switch the page
evidence and AI-reviewer layers on and off.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import anthropic
import streamlit as st
from dotenv import load_dotenv

import credibility
from credibility import score_band, score_url_detailed

load_dotenv()

# Claude Opus 5 is the most capable model. Swap to "claude-sonnet-5" or
# "claude-haiku-4-5" if you want to reduce cost while developing — note which
# one your submitted numbers used.
CHAT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Web search can return 20+ results per turn, and every displayed source is
# scored — one API call each when the LLM layer is on. Cap what we show.
MAX_SOURCES = 6

SYSTEM_PROMPT = """You are a research assistant for a graduate data science course.

Answer using the sources available to you and cite them. Be direct and concise.
When the evidence is thin or the sources disagree, say so plainly rather than
smoothing it over. Never invent a source or a URL."""


# -----------------------------------------------------------------------------
# Optional Langfuse tracing
# -----------------------------------------------------------------------------
# Tracing is a nice-to-have, not a requirement. If the Langfuse keys are absent
# we fall back to a no-op decorator so the app still runs on a fresh clone.
# This is why you can start working before configuring anything but the API key.
try:
    from langfuse import get_client, observe

    _langfuse = get_client()
    TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except Exception:
    TRACING_ENABLED = False
    _langfuse = None

    def observe(*_args, **_kwargs):  # type: ignore[misc]
        """No-op stand-in for @observe when Langfuse is not configured."""
        def decorator(fn):
            return fn
        return decorator


def search_serpapi(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Search Google via SerpAPI and return the organic results.

    This is optional context on top of Claude's own web search — it gives you a
    second, independently-retrieved set of URLs to score, which is useful when
    comparing how your scorer treats different kinds of source.
    """
    from serpapi import GoogleSearch

    search = GoogleSearch({"q": query, "api_key": api_key})
    return search.get_dict().get("organic_results", [])


# ---------------------------------------------------------------------------
# ⚠️  NEEDS YOUR OWN API KEY — AND SHIPPED UNVERIFIED
# ---------------------------------------------------------------------------
# REQUIRES A KEY. The chat does not work without ANTHROPIC_API_KEY in `.env`;
# the sidebar shows a red mark when it is missing. Get one at
# https://console.anthropic.com/. Calls are billed to you. The URL scorer in the
# sidebar, the tests, and evaluate.py all work without a key.
#
# VERIFIED LIVE — after a real bug was found here. The first live run returned
# ZERO sources, because this function originally read citations off the text
# blocks. `web_search_20260209` does not put them there: it returns them in
# `web_search_tool_result` blocks, and `block.citations` is None. The code below
# now reads both, and a live run yields six sources including the actual
# arXiv link for "Attention Is All You Need".
#
# The lesson is worth more than the fix: an API that returns an empty list where
# you expected data fails silently. Nothing crashed, no error was logged, the
# app just quietly showed no sources at all.
# ---------------------------------------------------------------------------
@observe()
def ask_claude(messages: List[Dict[str, str]], user: str, email: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Send the conversation to Claude and return the answer plus its citations.

    Where the sources come from: the `web_search_20260209` tool returns them in
    `web_search_tool_result` blocks, NOT as citation metadata on the text blocks.
    That is worth knowing — the obvious implementation reads `block.citations`,
    finds it empty, and silently shows no sources at all, which is exactly the
    bug this function was shipped with until it was run against the live API.

    :return: (answer_text, [{"url": ..., "title": ...}, ...])
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
    )

    # Claude can decline a request. Check before reading content, which is empty
    # or partial on a refusal.
    if response.stop_reason == "refusal":
        return ("I can't help with that request.", [])

    answer = ""
    citations: List[Dict[str, str]] = []
    seen: set = set()

    for block in response.content:
        if block.type == "text":
            answer += block.text
            # Some configurations attach citations directly to text blocks.
            # web_search_20260209 does NOT — see the branch below — but keep this
            # path so the app still works if that changes or you enable document
            # citations.
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(citation, "title", "") or url})

        elif block.type == "web_search_tool_result":
            # This is where the sources actually are. Each successful result block
            # holds a list of `web_search_result` items with .url and .title.
            # On failure `.content` is a single error object rather than a list,
            # so check the type before iterating.
            results = getattr(block, "content", None)
            if not isinstance(results, list):
                continue
            for item in results:
                url = getattr(item, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(item, "title", "") or url})

    # A single turn can return twenty-odd results across several searches, and the
    # app scores every one of them — which with the LLM layer on is one API call
    # each. Cap it: the first few are the ones the model actually leaned on.
    citations = citations[:MAX_SOURCES]

    if TRACING_ENABLED and _langfuse is not None:
        _langfuse.update_current_trace(
            input=messages[-1]["content"] if messages else "",
            output=answer,
            user_id=user,
            session_id=session_id,
            tags=["cs676", "project-1"],
            metadata={"email": email, "citations": len(citations)},
        )

    return answer, citations


# -----------------------------------------------------------------------------
# Scoring + rendering
# -----------------------------------------------------------------------------
# Scoring can take a few seconds per URL when the evidence layer reads pages,
# so every source in a turn is scored in parallel, and results are cached
# across Streamlit reruns (replaying the chat history must not refetch pages).
# The layout keeps one line per source; detail lives in a collapsed expander.

@st.cache_data(show_spinner=False, ttl=3600)
def cached_score(url: str, use_llm: bool, use_network: bool) -> Dict[str, Any]:
    """Memoised wrapper so reruns and repeated citations cost nothing."""
    return score_url_detailed(url, use_llm=None if use_llm else False, use_network=use_network)


def score_many(urls: List[str], use_llm: bool, use_network: bool) -> Dict[str, Dict[str, Any]]:
    """Score several URLs concurrently; one slow site cannot stall the others."""
    unique = list(dict.fromkeys(u for u in urls if u))
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(unique)))) as pool:
        results = list(pool.map(lambda u: cached_score(u, use_llm, use_network), unique))
    return dict(zip(unique, results))


ICONS = {"+": "✅", "-": "⚠️", "i": "ℹ️"}


def render_source(index: int, title: str, url: str, result: Dict[str, Any], snippet: str = "") -> None:
    """
    Render one source: title, coloured chip with the score and its range, and
    the source type on one line; the reasons sit in a collapsed expander so a
    list of six sources stays scannable.
    """
    label, colour = score_band(result["score"])
    rng = f"{result['low']:.2f}–{result['high']:.2f}"
    st.markdown(
        f"**{index}. [{title}]({url})**  \n"
        f":{colour}-badge[● {result['score']:.2f} {label}] "
        f":gray-badge[{result['source_type']}] "
        f"<span style='color:gray;font-size:0.8em'>likely {rng}</span>",
        unsafe_allow_html=True,
    )
    if snippet:
        st.caption(snippet)
    with st.expander("Why this score?"):
        st.write(result["explanation"])
        for item in result.get("evidence", []):
            st.markdown(f"{ICONS.get(item['direction'], '•')} {item['text']}")
        st.caption("Layers used: " + ", ".join(result.get("layers", [])) +
                   " · range is a 90% bootstrap interval")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CS676 — Credibility Chatbot", page_icon="🔍")
st.title("🔍 Credibility-Scored Research Assistant")

with st.sidebar:
    st.subheader("Session")
    user = st.text_input("Name", value="student")
    email = st.text_input("Email", value="student@pace.edu")
    session_id = f"{user}_{email}"

    st.divider()
    use_serpapi = st.checkbox("Also search with SerpAPI", value=False)
    st.caption("**Credibility scoring**")
    use_network = st.checkbox("Read pages & check scholarly databases", value=credibility._network_default(),
                              help="Fetches each source page and queries OpenAlex/Crossref for retractions "
                                   "and citations. Slower; turn off to score from the URL alone.")
    use_llm_scoring = st.checkbox("Add AI reviewer to scores", value=False,
                                  disabled=not os.getenv("ANTHROPIC_API_KEY"),
                                  help="One extra Claude call per source (billed to your key).")

    st.divider()
    st.caption("**Status**")
    st.caption(("✅" if os.getenv("ANTHROPIC_API_KEY") else "❌") + " Anthropic API key")
    st.caption(("✅" if os.getenv("SERPAPI_API_KEY") else "⬜") + " SerpAPI key (optional)")
    st.caption(("✅" if TRACING_ENABLED else "⬜") + " Langfuse tracing (optional)")

    st.divider()
    st.caption("Score any URL directly:")
    probe = st.text_input("URL", placeholder="https://arxiv.org/abs/1706.03762")
    if probe:
        with st.spinner("Scoring..."):
            probe_result = cached_score(probe.strip(), use_llm_scoring, use_network)
        probe_label, probe_colour = score_band(probe_result["score"])
        st.markdown(f":{probe_colour}-badge[**{probe_result['score']:.2f} — {probe_label}**] "
                    f":gray-badge[{probe_result['source_type']}]")
        st.caption(probe_result["explanation"])
        for item in probe_result.get("evidence", []):
            st.caption(f"{ICONS.get(item['direction'], '•')} {item['text']}")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and add your key. "
               "The URL scorer in the sidebar still works without one.")

# A batch scorer in the main pane: works with no API key, which makes it the
# fallback for a live demo, and shows exactly how chat sources are rendered.
with st.expander("Score a list of sources (no API key needed)", expanded=not os.getenv("ANTHROPIC_API_KEY")):
    batch = st.text_area("One URL per line", height=110, placeholder=(
        "https://www.pnas.org/doi/10.1073/pnas.2020123118\n"
        "https://arxiv.org/abs/1706.03762\n"
        "https://health-truth-daily.info/miracle-cure-doctors-hate"))
    batch_urls = [u.strip() for u in batch.splitlines() if u.strip()][:20]
    if batch_urls:
        with st.spinner("Scoring..."):
            batch_scores = score_many(batch_urls, use_llm_scoring, use_network)
        for i, u in enumerate(batch_urls, 1):
            render_source(i, u[:70] + ("…" if len(u) > 70 else ""), u, batch_scores[u])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            scored = score_many([x["url"] for x in sources], use_llm_scoring, use_network)
            for i, source in enumerate(sources, 1):
                render_source(i, source["title"], source["url"], scored[source["url"]], source.get("snippet", ""))

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the request separately from the stored history. Search context is
    # useful for this turn only — writing it back into session_state would
    # re-send it on every later turn and inflate the conversation.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    serp_sources: List[Dict[str, str]] = []

    if use_serpapi and os.getenv("SERPAPI_API_KEY"):
        try:
            results = search_serpapi(prompt, os.getenv("SERPAPI_API_KEY"))[:5]
            if results:
                context = "\n\nSearch results for reference:\n"
                for r in results:
                    title = r.get("title", "Untitled")
                    link = r.get("link", "")
                    snippet = r.get("snippet", "")
                    serp_sources.append({"title": title, "url": link, "snippet": snippet})
                    context += f"- {title} ({link})\n  {snippet}\n"
                api_messages[-1] = {"role": "user", "content": prompt + context}
        except Exception as e:
            st.warning(f"SerpAPI search failed: {e}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, citations = ask_claude(api_messages, user, email, session_id)
            except Exception as e:
                answer, citations = f"Error: {e}", []

        st.markdown(answer)

        # Merge Claude's own citations with any SerpAPI results, dropping dupes.
        sources: List[Dict[str, str]] = []
        seen_urls: set = set()
        for source in citations + serp_sources:
            if source["url"] and source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                sources.append(source)

        if sources:
            st.divider()
            st.caption(f"**{len(sources)} source(s), scored by `credibility.score_url`**")
            with st.spinner("Checking sources..."):
                scored = score_many([x["url"] for x in sources], use_llm_scoring, use_network)
            for i, source in enumerate(sources, 1):
                render_source(i, source["title"], source["url"], scored[source["url"]], source.get("snippet", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
