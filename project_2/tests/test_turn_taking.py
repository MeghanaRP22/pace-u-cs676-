"""
Tests for the smarter conversation loop: who speaks next, staying silent,
ending early, surviving a failed call, and the transcript window.
"""

import pytest

from personaforge import PersonaSpec, ScriptedLLM, agent_from_spec, run_conversation
from personaforge.conversation import (
    ENDED_BY_AGENT,
    ENDED_ERROR,
    ENDED_MAX_TURNS,
    ENDED_SILENCE,
    Turn,
    parse_control,
    save_transcript,
)


class PerAgentLLM:
    """Each agent gets its own script, keyed by name in the system prompt."""

    def __init__(self, scripts):
        self.scripts = {name: list(lines) for name, lines in scripts.items()}
        self.calls = []

    def complete(self, system, messages, max_tokens=2048):
        self.calls.append({"system": system, "messages": list(messages)})
        for name, lines in self.scripts.items():
            if f"Speak only as {name}." in system:
                return lines.pop(0) if lines else "[PASS]"
        raise AssertionError("unknown agent")


def cast(llm, *people):
    return [agent_from_spec(PersonaSpec(name=n, role=r, summary="", body=n), llm) for n, r in people]


# -- control tokens -----------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Hello there.", ("Hello there.", False, False)),
        ("[PASS]", ("", True, False)),
        ("[pass].", ("", True, False)),
        ("That settles it. [END]", ("That settles it.", False, True)),
        ("Goodbye.\n[end]", ("Goodbye.", False, True)),
        ("", ("", True, False)),
    ],
)
def test_parse_control(raw, expected):
    assert parse_control(raw) == expected


# -- who speaks next ----------------------------------------------------------


def test_addressed_participant_speaks_next():
    llm = PerAgentLLM({
        "Ann Lee": ["Bo, what do you think?", "ok"],
        "Bo Kim": ["I agree.", "fine"],
        "Cy Diaz": ["Me too.", "sure"],
    })
    agents = cast(llm, ("Ann Lee", "doctor"), ("Cy Diaz", "nurse"), ("Bo Kim", "patient"))
    convo = run_conversation(agents, topic="x", turns=2)

    # Round-robin would have picked Cy second; Ann addressed Bo by name.
    assert [t.speaker for t in convo.turns] == ["Ann Lee", "Bo Kim"]


def test_role_counts_as_addressing():
    llm = PerAgentLLM({"Ann Lee": ["Nurse, the chart please."], "Bo Kim": ["x"], "Cy Diaz": ["Here."]})
    agents = cast(llm, ("Ann Lee", "doctor"), ("Bo Kim", "patient"), ("Cy Diaz", "nurse"))
    assert run_conversation(agents, topic="x", turns=2).turns[1].speaker == "Cy Diaz"


def test_unaddressed_goes_to_whoever_was_quiet_longest():
    llm = PerAgentLLM({"A One": ["a1", "a2"], "B Two": ["b1", "b2"], "C Three": ["c1", "c2"]})
    agents = cast(llm, ("A One", "x"), ("B Two", "y"), ("C Three", "z"))
    convo = run_conversation(agents, topic="x", turns=6)
    assert [t.speaker for t in convo.turns] == ["A One", "B Two", "C Three"] * 2


# -- silence and ending -------------------------------------------------------


def test_pass_is_not_a_line_and_is_counted():
    llm = PerAgentLLM({"Ann Lee": ["one", "two"], "Bo Kim": ["[PASS]", "three"]})
    convo = run_conversation(cast(llm, ("Ann Lee", "d"), ("Bo Kim", "p")), topic="x", turns=3)

    assert all("PASS" not in t.text for t in convo.turns)
    assert convo.passes == {"Bo Kim": 1}


def test_everyone_silent_ends_the_conversation():
    llm = PerAgentLLM({"Ann Lee": ["hello"], "Bo Kim": []})  # both pass after the first line
    convo = run_conversation(cast(llm, ("Ann Lee", "d"), ("Bo Kim", "p")), topic="x", turns=10)

    assert convo.ended_by == ENDED_SILENCE
    assert len(convo.turns) == 1


def test_end_token_closes_once_everyone_has_spoken():
    llm = PerAgentLLM({"Ann Lee": ["Hi.", "Bye then."], "Bo Kim": ["Got what I needed. [END]"]})
    convo = run_conversation(cast(llm, ("Ann Lee", "d"), ("Bo Kim", "p")), topic="x", turns=10)

    assert convo.ended_by == ENDED_BY_AGENT
    assert [t.text for t in convo.turns] == ["Hi.", "Got what I needed."]


def test_end_token_from_first_speaker_alone_is_ignored():
    llm = PerAgentLLM({"Ann Lee": ["Done. [END]", "Still here."], "Bo Kim": ["Wait!"]})
    convo = run_conversation(cast(llm, ("Ann Lee", "d"), ("Bo Kim", "p")), topic="x", turns=3)

    assert convo.ended_by == ENDED_MAX_TURNS
    assert len(convo.turns) == 3


def test_turns_must_be_positive():
    llm = ScriptedLLM()
    with pytest.raises(ValueError):
        run_conversation(cast(llm, ("A", "x"), ("B", "y")), topic="x", turns=0)


# -- robustness ---------------------------------------------------------------


class FailsOnCall:
    def __init__(self, fail_at):
        self.fail_at, self.n = fail_at, 0

    def complete(self, system, messages, max_tokens=2048):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("API down")
        return f"line {self.n}"


def test_failure_mid_conversation_keeps_what_was_said(tmp_path):
    convo = run_conversation(cast(FailsOnCall(3), ("Ann Lee", "d"), ("Bo Kim", "p")), topic="x", turns=6)

    assert [t.text for t in convo.turns] == ["line 1", "line 2"]
    assert convo.ended_by == ENDED_ERROR
    assert "API down" in convo.error
    # The partial transcript still saves, with the error noted.
    assert "API down" in save_transcript(convo, tmp_path).read_text()


def test_transcripts_never_overwrite_each_other(tmp_path):
    llm = ScriptedLLM(replies=["x", "y"])
    agents = cast(llm, ("A", "x"), ("B", "y"))
    paths = {save_transcript(run_conversation(agents, topic="same", turns=2), tmp_path) for _ in range(3)}
    assert len(paths) == 3


# -- what an agent sees -------------------------------------------------------


def test_long_transcripts_are_windowed_but_keep_the_scene():
    llm = ScriptedLLM(replies=["ok"])
    agent = agent_from_spec(PersonaSpec(name="Ann", role="d", summary="", body="b"), llm)
    agent.window = 5
    history = [Turn("Scene", "THE_SCENE")] + [Turn("Bo" if i % 2 else "Ann", f"line {i}") for i in range(30)]
    agent.respond(history)

    sent = llm.calls[0]["messages"]
    assert "THE_SCENE" in sent[0]["content"]
    assert len([m for m in sent if "line" in m["content"]]) == 4


def test_own_line_last_gets_an_explicit_silence():
    llm = ScriptedLLM(replies=["ok"])
    agent = agent_from_spec(PersonaSpec(name="Ann", role="d", summary="", body="b"), llm)
    agent.respond([Turn("Scene", "s"), Turn("Ann", "Anyone?")])

    last = llm.calls[0]["messages"][-1]
    assert last["role"] == "user" and "silence" in last["content"]


def test_system_prompt_explains_pass_and_end():
    agent = agent_from_spec(PersonaSpec(name="Ann", role="d", summary="", body="b"), ScriptedLLM())
    prompt = agent.system_prompt()
    assert "[PASS]" in prompt and "[END]" in prompt
