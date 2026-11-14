"""
Tests for ClaudeLLM (against a fake client — no network, no cost), DemoLLM's
offline tool loop, and the CLI.

The fake client returns objects shaped like the SDK's, so these check our side
of the contract: how we build requests, read tool_use blocks, count usage, and
turn API failures into messages a student can act on.
"""

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from personaforge import ClaudeLLM, DemoLLM, LLMError, Orchestrator
from personaforge import cli, ui
from personaforge.llm import supports_tools

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def block(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def response(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )


class FakeClient:
    def __init__(self, result):
        self.result, self.requests = result, []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def claude(result):
    return ClaudeLLM(model="test-model", _client=FakeClient(result))


# -- ClaudeLLM ----------------------------------------------------------------


def test_complete_joins_text_and_counts_usage():
    llm = claude(response(block("text", text="Hello "), block("text", text="there.")))
    assert llm.complete("sys", [{"role": "user", "content": "hi"}]) == "Hello there."
    assert llm.usage == {"calls": 1, "input_tokens": 100, "output_tokens": 20}
    assert llm._client.requests[0]["model"] == "test-model"


def test_refusal_is_reported_not_crashed():
    llm = claude(response(stop_reason="refusal"))
    assert llm.complete("sys", [{"role": "user", "content": "x"}]) == "[declined to respond]"


def test_complete_with_tools_reads_tool_use_blocks():
    llm = claude(
        response(
            block("text", text="On it."),
            block("tool_use", id="toolu_1", name="create_persona", input={"description": "a nurse"}),
            stop_reason="tool_use",
        )
    )
    turn = llm.complete_with_tools("sys", [{"role": "user", "content": "x"}], tools=[{"name": "t"}])

    assert turn.text == "On it."
    assert turn.tool_calls[0].name == "create_persona"
    assert turn.tool_calls[0].input == {"description": "a nurse"}
    # The echo must be plain dicts in API shape.
    assert turn.content[1] == {"type": "tool_use", "id": "toolu_1", "name": "create_persona", "input": {"description": "a nurse"}}
    assert llm._client.requests[0]["tools"] == [{"name": "t"}]


def test_tool_refusal_becomes_text():
    llm = claude(response(stop_reason="refusal"))
    assert llm.complete_with_tools("s", [], tools=[]).text == "[declined to respond]"


@pytest.mark.parametrize(
    "error, expected",
    [
        (anthropic.APIConnectionError(request=REQUEST), "network"),
        (anthropic.AuthenticationError("bad", response=httpx.Response(401, request=REQUEST), body=None), "key was rejected"),
        (anthropic.NotFoundError("no", response=httpx.Response(404, request=REQUEST), body=None), "--model"),
        (anthropic.RateLimitError("slow", response=httpx.Response(429, request=REQUEST), body=None), "Rate limited"),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQUEST), body=None), "500"),
    ],
)
def test_api_errors_become_actionable_llm_errors(error, expected):
    with pytest.raises(LLMError, match=expected):
        claude(error).complete("s", [{"role": "user", "content": "x"}])


def test_missing_key_raises_llm_error_with_instructions(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="--offline"):
        ClaudeLLM().complete("s", [{"role": "user", "content": "x"}])


def test_orchestrator_turns_api_failure_into_llm_error(tmp_path):
    llm = claude(anthropic.APIConnectionError(request=REQUEST))
    with pytest.raises(LLMError):
        Orchestrator(llm, tmp_path).handle("hello")


# -- DemoLLM drives the real tool loop offline --------------------------------


def test_demo_llm_supports_tools_and_runs_a_whole_session(tmp_path):
    llm = DemoLLM()
    assert supports_tools(llm)
    orch = Orchestrator(llm, tmp_path)

    created = orch.handle("make it a patient and a doctor, then have them talk about the scan")
    assert created.conversation is not None
    assert created.evaluation is not None and created.evaluation.goal_met == 3
    # The demo dialogue ends itself with [END], exercising early ending offline.
    assert created.conversation.ended_by == "a participant closed it"
    assert "offline" in orch.handle("good morning").reply.lower()


def test_demo_llm_edit_returns_the_persona_unchanged(tmp_path):
    orch = Orchestrator(DemoLLM(), tmp_path)
    orch.handle("create a persona doctor")
    before = (tmp_path / "doctor.md").read_text()
    orch.handle("make the doctor more skeptical")
    assert (tmp_path / "doctor.md").read_text() == before


# -- CLI ----------------------------------------------------------------------


def run_cli(monkeypatch, tmp_path, lines, *flags):
    feed = iter(lines)

    def fake_prompt():
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr(ui, "user_prompt", fake_prompt)
    with ui.console.capture() as captured:
        code = cli.main(["--offline", "--persona-dir", str(tmp_path), *flags])
    return code, captured.get()


def test_cli_offline_session(monkeypatch, tmp_path):
    code, out = run_cli(
        monkeypatch,
        tmp_path,
        ["", "/help", "/cost", "/reset", "create a persona patient", "create a persona doctor",
         "have them talk about the results", "/clear", "/quit"],
    )
    assert code == 0
    assert "Slash commands" in out
    assert "No usage to report" in out
    assert "Observer" in out
    assert (tmp_path / "patient.md").exists()


def test_cli_exits_cleanly_on_eof(monkeypatch, tmp_path):
    code, out = run_cli(monkeypatch, tmp_path, [])
    assert code == 0 and "bye" in out


def test_cli_reports_errors_without_dying(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise LLMError("network down — try again")

    monkeypatch.setattr(Orchestrator, "handle", boom)
    code, out = run_cli(monkeypatch, tmp_path, ["hello", "/quit"])
    assert code == 0 and "network down" in out


def test_cli_flags(monkeypatch, tmp_path):
    code, _ = run_cli(monkeypatch, tmp_path, ["create a persona a", "create a persona b", "have them talk"],
                      "--no-observer", "--no-memory")
    assert code == 0
    assert not (tmp_path / "memory").exists()


def test_usage_rendering():
    with ui.console.capture() as captured:
        ui.usage({"calls": 3, "input_tokens": 1500, "output_tokens": 200})
    assert "3 API call(s)" in captured.get() and "1,500" in captured.get()
