"""
streamlit_app.py — a browser chat over the personaforge package.

    uv run streamlit run deploy/streamlit_app.py

This file is presentation only, the web twin of cli.py: every decision is made
by personaforge.Orchestrator. If logic ever needs to be added here, it belongs
in the package instead.

Deployment notes (see README "Deploying"):
  * It lives in deploy/ next to its own requirements.txt because Streamlit
    Community Cloud reads uv.lock before requirements.txt, but checks the
    entrypoint's folder before the repository root.
  * The host never installs personaforge, so we put ../src on the path.
  * Each browser session gets its own temporary persona folder. The server's
    disk is shared, and visitors should not see each other's characters.
  * Live mode spends the deployer's API key, so each session is capped.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st  # noqa: E402  (must follow the path fix above)

from personaforge import Orchestrator, PersonaForgeError, __version__, default_llm, list_personas  # noqa: E402

WEB_MODEL = "claude-haiku-4-5"  # cheapest sensible default for a public demo
MAX_LIVE_REQUESTS = 15  # per browser session, to protect the deployer's key


def _key_available() -> bool:
    """Streamlit exposes root-level secrets as environment variables."""
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _new_session(offline: bool) -> None:
    """Fresh orchestrator, fresh private persona folder, empty chat."""
    folder = Path(tempfile.mkdtemp(prefix="personaforge-"))
    llm = default_llm(offline=offline, model=WEB_MODEL)
    st.session_state.update(
        orchestrator=Orchestrator(llm=llm, persona_dir=folder),
        offline=offline,
        log=[],
        live_requests=0,
    )


def _render(entry: dict) -> None:
    """Draw one remembered chat entry."""
    if entry["kind"] == "user":
        st.chat_message("user").markdown(entry["text"])
    elif entry["kind"] == "line":
        st.chat_message("assistant", avatar="🎭").markdown(f"**{entry['speaker']}:** {entry['text']}")
    elif entry["kind"] == "details":
        with st.expander("Details", expanded=True):
            st.markdown(entry["text"])
    elif entry["kind"] == "observer":
        st.info(entry["text"], icon="🔎")
    else:
        st.chat_message("assistant", avatar="🎬").markdown(entry["text"])


st.set_page_config(page_title="PersonaForge", page_icon="🎭")
st.title("🎭 PersonaForge")
st.caption(f"Agent-to-agent persona simulation · v{__version__}")

if "orchestrator" not in st.session_state:
    _new_session(offline=not _key_available())

with st.sidebar:
    st.subheader("Mode")
    if _key_available():
        offline = st.toggle("Offline (scripted, free)", value=st.session_state.offline)
        if offline != st.session_state.offline:
            _new_session(offline)
    else:
        st.warning("No API key configured — running offline with scripted replies.")
    if not st.session_state.offline:
        st.caption(f"Live · {WEB_MODEL} · {st.session_state.live_requests}/{MAX_LIVE_REQUESTS} requests used")

    st.subheader("Personas")
    orch: Orchestrator = st.session_state.orchestrator
    specs = list_personas(orch.persona_dir)
    for spec in specs:
        st.markdown(f"**{spec.name}** · {spec.role}  \n<small>{spec.summary}</small>", unsafe_allow_html=True)
    if not specs:
        st.caption("None yet.")

    transcripts = sorted((orch.persona_dir / "conversations").glob("*.md"))
    if transcripts:
        st.download_button(
            "Download latest transcript",
            transcripts[-1].read_text(encoding="utf-8"),
            file_name=transcripts[-1].name,
        )
    if st.button("Start over"):
        _new_session(st.session_state.offline)
        st.rerun()

    st.divider()
    st.markdown(
        "Try:\n"
        "- *a patient with chronic back pain who distrusts doctors, and a blunt internist running late*\n"
        "- *have them talk about the MRI results*\n"
        "- *make the doctor more patient*"
    )

for entry in st.session_state.log:
    _render(entry)

prompt = st.chat_input("Describe who to create, or what they should talk about")
if prompt:
    log = st.session_state.log

    def remember(entry: dict) -> None:
        log.append(entry)
        _render(entry)

    remember({"kind": "user", "text": prompt})

    if not st.session_state.offline and st.session_state.live_requests >= MAX_LIVE_REQUESTS:
        remember({"kind": "reply", "text": "This session has used its live requests. Press **Start over** in offline mode, or run it locally with your own key."})
    else:
        st.session_state.live_requests += 0 if st.session_state.offline else 1
        with st.spinner("Working…"):
            try:
                result = st.session_state.orchestrator.handle(
                    prompt,
                    on_turn=lambda turn: remember({"kind": "line", "speaker": turn.speaker, "text": turn.text}),
                )
            except PersonaForgeError as exc:
                remember({"kind": "reply", "text": f"⚠️ {exc}"})
            else:
                if result.evaluation is not None:
                    ev = result.evaluation
                    score = f"{ev.goal_met}/5" if ev.goal_met is not None else "unscored"
                    flags = "".join(f"\n- ⚑ {f}" for f in ev.all_flags)
                    remember({"kind": "observer", "text": f"**Observer · goal {score}** — {ev.summary}{flags}"})
                for block in result.details:
                    remember({"kind": "details", "text": block})
                remember({"kind": "reply", "text": result.reply})
    st.rerun()
