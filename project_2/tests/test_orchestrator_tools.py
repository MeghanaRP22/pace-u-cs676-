"""
Tests for the tool-use orchestrator — the headline extension.

ScriptedToolLLM plays the part of Claude: each test scripts which tools "the
model" chooses, and the assertions check that the orchestrator runs them,
reports back correctly, and never sends the API a malformed history.
"""

from personaforge import Orchestrator, PersonaSpec, ScriptedToolLLM, ToolTurn, save_persona
from personaforge.orchestrator import HISTORY_LIMIT, MAX_TOOL_ROUNDS, MAX_TURNS_LIMIT, TOOLS

PERSONA_MD = """---
name: {name}
role: {role}
summary: {role} for tests
---

# {name}

## Background
Test fixture.

## Personality
- Calm

## How they speak
- Briefly
"""


def persona(name, role):
    return PERSONA_MD.format(name=name, role=role)


def seed(tmp_path, *people):
    for name, role in people:
        save_persona(PersonaSpec(name=name, role=role, summary="s", body="## Background\nb\n## Personality\np"), tmp_path)


# -- the loop -----------------------------------------------------------------


def test_tools_are_offered_to_the_model(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.say("Hi.")])
    Orchestrator(llm, tmp_path).handle("hello")

    offered = {t["name"] for t in llm.tool_calls_seen[0]["tools"]}
    assert offered == {"create_persona", "list_personas", "show_persona", "edit_persona", "start_conversation"}


def test_text_reply_without_tools_is_returned_as_is(tmp_path):
    """A clarifying question is just a reply with no tool calls."""
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.say("What kind of persona — a patient, a doctor?")])
    result = Orchestrator(llm, tmp_path).handle("create a persona")

    assert result.reply.startswith("What kind of persona")
    assert list(tmp_path.glob("*.md")) == []  # nothing was invented


def test_several_tool_calls_in_one_turn(tmp_path):
    """'two doctors and a nurse' is three create_persona calls in one response."""
    llm = ScriptedToolLLM(
        tool_turns=[
            ToolTurn.use(
                ("create_persona", {"description": "a doctor"}),
                ("create_persona", {"description": "another doctor"}),
                ("create_persona", {"description": "a nurse"}),
            ),
            ToolTurn.say("Created all three."),
        ],
        replies=[persona("Ann Lee", "doctor"), persona("Raj Patel", "doctor"), persona("Bo Kim", "nurse")],
    )
    result = Orchestrator(llm, tmp_path).handle("make two doctors and a nurse")

    assert sorted(p.stem for p in tmp_path.glob("*.md")) == ["ann-lee", "bo-kim", "raj-patel"]
    assert result.reply == "Created all three."
    assert len(result.events) == 3


def test_tool_results_are_sent_back_with_matching_ids(tmp_path):
    turn = ToolTurn.use(("list_personas", {}))
    llm = ScriptedToolLLM(tool_turns=[turn, ToolTurn.say("Nobody yet.")])
    Orchestrator(llm, tmp_path).handle("who do I have?")

    second_request = llm.tool_calls_seen[1]["messages"]
    assistant, results = second_request[-2], second_request[-1]
    assert assistant["role"] == "assistant" and assistant["content"] == turn.content
    assert results["role"] == "user"
    assert results["content"][0]["type"] == "tool_result"
    assert results["content"][0]["tool_use_id"] == turn.tool_calls[0].id


def test_multi_step_request_runs_in_order(tmp_path):
    """Create two personas, then talk — the skeleton's regexes could only do one."""
    llm = ScriptedToolLLM(
        tool_turns=[
            ToolTurn.use(("create_persona", {"description": "patient"}), ("create_persona", {"description": "doctor"})),
            ToolTurn.use(("start_conversation", {"topic": "the scan", "max_turns": 2})),
            ToolTurn.say("Done."),
        ],
        replies=[persona("Maria Delgado", "patient"), persona("Sam Reyes", "doctor"), "a line"],
    )
    result = Orchestrator(llm, tmp_path, use_observer=False).handle("patient and doctor, then have them talk")

    assert result.conversation is not None
    # "have them" means the two just created, in creation order.
    assert result.conversation.participants == ["Maria Delgado", "Sam Reyes"]


def test_unknown_persona_becomes_an_error_result_not_a_crash(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "nurse"))
    llm = ScriptedToolLLM(
        tool_turns=[
            ToolTurn.use(("start_conversation", {"topic": "x", "participants": ["Zaphod", "Bo"]})),
            ToolTurn.say("I couldn't find Zaphod — did you mean Ann?"),
        ]
    )
    result = Orchestrator(llm, tmp_path).handle("have Zaphod and Bo talk")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "Zaphod" in tool_result["content"] and "Ann Lee" in tool_result["content"]
    assert result.conversation is None


def test_ambiguous_participant_is_reported_to_the_model(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Raj Patel", "doctor"), ("Bo Kim", "nurse"))
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.use(("show_persona", {"name": "doctor"})), ToolTurn.say("Which one?")])
    Orchestrator(llm, tmp_path).handle("show me the doctor")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "more than one" in tool_result["content"]


def test_more_than_two_personas_requires_naming_participants(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "nurse"), ("Cy Diaz", "patient"))
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x"})), ToolTurn.say("Who?")])
    Orchestrator(llm, tmp_path).handle("have them talk")

    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "say who should talk" in tool_result["content"]


def test_unknown_tool_name_is_an_error_result(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.use(("delete_everything", {})), ToolTurn.say("Sorry.")])
    Orchestrator(llm, tmp_path).handle("delete everything")

    assert "Unknown tool" in llm.tool_calls_seen[1]["messages"][-1]["content"][0]["content"]


def test_runaway_tool_loop_is_capped(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.use(("list_personas", {}))] * (MAX_TOOL_ROUNDS + 5))
    result = Orchestrator(llm, tmp_path).handle("list forever")

    assert len(llm.tool_calls_seen) == MAX_TOOL_ROUNDS
    assert "stopped after several steps" in result.reply


def test_empty_create_description_asks_instead_of_inventing(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.use(("create_persona", {"description": ""})), ToolTurn.say("Who?")])
    Orchestrator(llm, tmp_path).handle("create a persona")

    assert llm.calls == []  # the author model was never called
    assert llm.tool_calls_seen[1]["messages"][-1]["content"][0]["is_error"] is True


# -- tools in detail ----------------------------------------------------------


def test_max_turns_is_clamped(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "nurse"))
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x", "max_turns": 500})), ToolTurn.say("ok")],
        replies=["a different line {}".format(i) for i in range(50)],
    )
    result = Orchestrator(llm, tmp_path, use_observer=False).handle("talk a lot")

    assert len(result.conversation.turns) == MAX_TURNS_LIMIT


def test_bad_max_turns_falls_back_to_default(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "nurse"))
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x", "max_turns": "lots"})), ToolTurn.say("ok")],
        replies=[f"line {i}" for i in range(20)],
    )
    result = Orchestrator(llm, tmp_path, use_observer=False).handle("talk")
    assert len(result.conversation.turns) == 6


def test_list_and_show_are_displayed_verbatim(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"))
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("list_personas", {}), ("show_persona", {"name": "ann"})), ToolTurn.say("There.")]
    )
    result = Orchestrator(llm, tmp_path).handle("list and show ann")

    assert len(result.details) == 2
    assert "Ann Lee" in result.details[0]
    assert "## Background" in result.details[1]


def test_edit_persona_rewrites_the_file_and_keeps_the_name(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"))
    revised = persona("Someone Else", "doctor").replace("Calm", "Deeply skeptical")
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("edit_persona", {"name": "ann", "instruction": "more skeptical"})), ToolTurn.say("ok")],
        replies=[revised],
    )
    Orchestrator(llm, tmp_path).handle("make ann more skeptical")

    text = (tmp_path / "ann-lee.md").read_text()
    assert "Deeply skeptical" in text
    assert "name: Ann Lee" in text
    assert len(list(tmp_path.glob("*.md"))) == 1  # edited in place, not duplicated


def test_edit_with_malformed_revision_leaves_file_alone(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"))
    before = (tmp_path / "ann-lee.md").read_text()
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("edit_persona", {"name": "ann", "instruction": "x"})), ToolTurn.say("ok")],
        replies=["Sorry, I can't do that."],
    )
    Orchestrator(llm, tmp_path).handle("edit ann")

    assert (tmp_path / "ann-lee.md").read_text() == before
    assert llm.tool_calls_seen[1]["messages"][-1]["content"][0]["is_error"] is True


def test_conversation_is_observed_and_remembered(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "patient"))
    judge = '{"summary": "They agreed on a scan.", "goal_met": 4, "goal_reason": "Plan made.", "flags": []}'
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "the scan", "max_turns": 2})), ToolTurn.say("ok")],
        replies=["Hello Bo.", "Hello doctor.", judge],
    )
    result = Orchestrator(llm, tmp_path).handle("have them talk about the scan")

    assert result.evaluation is not None and result.evaluation.goal_met == 4
    assert "Observer" in next((tmp_path / "conversations").glob("*.md")).read_text()
    memory = (tmp_path / "memory" / "ann-lee.md").read_text()
    assert "Talked with Bo Kim about the scan" in memory
    assert "They agreed on a scan." in memory


def test_observer_and_memory_can_be_switched_off(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "patient"))
    llm = ScriptedToolLLM(
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x", "max_turns": 2})), ToolTurn.say("ok")],
        replies=["one", "two"],
    )
    result = Orchestrator(llm, tmp_path, use_observer=False, use_memory=False).handle("talk")

    assert result.evaluation is None
    assert len(llm.calls) == 2  # two lines of dialogue, no judge call
    assert not (tmp_path / "memory").exists()


def test_duplicate_names_are_disambiguated_in_conversation(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Ann Lee", "nurse"))
    llm = ScriptedToolLLM(
        tool_turns=[
            ToolTurn.use(("start_conversation", {"topic": "x", "participants": ["ann-lee", "ann-lee-2"], "max_turns": 2})),
            ToolTurn.say("ok"),
        ],
        replies=["one", "two"],
    )
    result = Orchestrator(llm, tmp_path, use_observer=False).handle("talk")

    assert result.conversation.participants == ["Ann Lee (ann-lee)", "Ann Lee (ann-lee-2)"]


# -- history ------------------------------------------------------------------


def test_history_is_trimmed_at_a_safe_boundary(tmp_path):
    llm = ScriptedToolLLM(
        tool_turns=[t for _ in range(HISTORY_LIMIT) for t in (ToolTurn.use(("list_personas", {})), ToolTurn.say("ok"))]
    )
    orch = Orchestrator(llm, tmp_path)
    for i in range(HISTORY_LIMIT):
        orch.handle(f"message {i}")

    assert len(orch.history) <= HISTORY_LIMIT
    first = orch.history[0]
    # Never start on a tool_result whose tool_use was cut off.
    assert first["role"] == "user" and isinstance(first["content"], str)


def test_reset_clears_history(tmp_path):
    llm = ScriptedToolLLM(tool_turns=[ToolTurn.say("hi")])
    orch = Orchestrator(llm, tmp_path)
    orch.handle("hello")
    orch.reset()
    assert orch.history == []


def test_tool_schemas_are_well_formed():
    for tool in TOOLS:
        assert set(tool) == {"name", "description", "input_schema"}
        assert tool["input_schema"]["type"] == "object"
        for required in tool["input_schema"].get("required", []):
            assert required in tool["input_schema"]["properties"]


# -- failures -----------------------------------------------------------------


class FlakyToolLLM(ScriptedToolLLM):
    """Dialogue works for `ok_calls` calls, then every complete() raises LLMError."""

    def __init__(self, ok_calls, **kwargs):
        super().__init__(**kwargs)
        self.ok_calls = ok_calls

    def complete(self, system, messages, max_tokens=2048):
        if len(self.calls) >= self.ok_calls:
            from personaforge import LLMError

            raise LLMError("network down")
        return super().complete(system, messages, max_tokens)


def test_api_failure_mid_conversation_keeps_partial_transcript(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "patient"))
    llm = FlakyToolLLM(
        ok_calls=1,
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x", "max_turns": 4})), ToolTurn.say("Partly.")],
        replies=["first line"],
    )
    result = Orchestrator(llm, tmp_path).handle("talk")

    assert [t.text for t in result.conversation.turns] == ["first line"]
    tool_result = llm.tool_calls_seen[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True and "network down" in tool_result["content"]
    assert any("wrote" in e for e in result.events)  # the partial transcript was still saved


def test_observer_failure_does_not_lose_the_conversation(tmp_path):
    seed(tmp_path, ("Ann Lee", "doctor"), ("Bo Kim", "patient"))
    llm = FlakyToolLLM(
        ok_calls=2,  # both lines succeed, the judge call fails
        tool_turns=[ToolTurn.use(("start_conversation", {"topic": "x", "max_turns": 2})), ToolTurn.say("ok")],
        replies=["one", "two"],
    )
    result = Orchestrator(llm, tmp_path).handle("talk")

    assert len(result.conversation.turns) == 2
    assert result.evaluation is None
    assert any("observer skipped" in e for e in result.events)
