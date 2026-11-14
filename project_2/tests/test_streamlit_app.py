"""
Smoke tests for the Streamlit front end, run headless with Streamlit's AppTest.

Offline only: the app falls back to DemoLLM when no key is set, so these cost
nothing. They check the wiring — the package does the real work and has its
own tests.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "deploy" / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_app_starts_offline_without_a_key(app):
    assert not app.exception
    assert app.session_state.offline is True
    assert any("No API key" in w.value for w in app.sidebar.warning)


def test_create_and_converse_in_the_browser(app):
    app.chat_input[0].set_value("create a persona patient").run()
    app.chat_input[0].set_value("create a persona doctor").run()
    app.chat_input[0].set_value("have them talk about the scan").run()

    assert not app.exception
    lines = [e for e in app.session_state.log if e["kind"] == "line"]
    assert len(lines) >= 2
    assert any(e["kind"] == "observer" for e in app.session_state.log)
    assert len(app.sidebar.get("download_button")) == 1


def test_each_session_has_its_own_folder(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    first, second = AppTest.from_file(APP), AppTest.from_file(APP)
    first.run()
    second.run()
    assert first.session_state.orchestrator.persona_dir != second.session_state.orchestrator.persona_dir
