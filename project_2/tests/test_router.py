"""
Tests for router.plan(), the pattern planner used as a fallback and offline.

These pin the three improvements over the skeleton's regexes: sequencing with
"then", counted personas, and a configurable turn count.
"""

from personaforge import Orchestrator, PersonaSpec, ScriptedLLM, save_persona
from personaforge.router import plan


def names(steps):
    return [name for name, _ in steps]


def test_plain_create():
    assert plan("create a persona patient with back pain") == [
        ("create_persona", {"description": "patient with back pain"})
    ]


def test_create_without_the_word_persona():
    assert plan("create a skeptical data scientist named Ada") == [
        ("create_persona", {"description": "skeptical data scientist named ada"})
    ]


def test_and_inside_a_description_is_not_a_second_persona():
    assert len(plan("create a persona doctor who is direct and busy")) == 1


def test_counts_expand_to_several_creates():
    steps = plan("make it two doctors and a nurse")
    assert names(steps) == ["create_persona"] * 3
    assert steps[0][1]["description"] == "doctor"
    assert steps[2][1]["description"] == "nurse"


def test_then_runs_both_steps_in_order():
    """The skeleton only ever listed here — pattern order decided meaning."""
    assert names(plan("list the personas then have them talk about lunch")) == ["list_personas", "start_conversation"]


def test_create_and_have_them_argue():
    steps = plan("make it two doctors and a nurse, and have them argue about the diagnosis")
    assert names(steps) == ["create_persona"] * 3 + ["start_conversation"]
    assert steps[-1][1]["topic"] == "the diagnosis"


def test_conversation_with_participants_topic_and_turns():
    (name, args), = plan("start a conversation between carol and bob about the chart for 10 turns")
    assert name == "start_conversation"
    assert args == {"topic": "the chart", "participants": ["carol", "bob"], "max_turns": 10}


def test_edit_and_show():
    assert plan("make the doctor more skeptical") == [
        ("edit_persona", {"name": "the doctor", "instruction": "more skeptical"})
    ]
    assert plan("show me the persona maria") == [("show_persona", {"name": "maria"})]


def test_making_a_new_persona_is_not_an_edit():
    assert names(plan("make a persona who is more skeptical than most")) == ["create_persona"]


def test_chat_produces_no_plan():
    assert plan("hello, what is this thing?") == []


def test_fallback_orchestrator_asks_when_create_is_empty(tmp_path):
    reply = Orchestrator(ScriptedLLM(), tmp_path).handle("create a persona").reply
    assert "what kind of persona" in reply.lower()


def test_fallback_runs_every_step(tmp_path):
    for name, role in [("Alice Adams", "doctor"), ("Bob Brown", "patient")]:
        save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)
    result = Orchestrator(ScriptedLLM(replies=["a line"]), tmp_path, use_observer=False).handle(
        "list personas then have them talk about lunch for 2 turns"
    )
    assert "Alice Adams" in result.reply
    assert len(result.conversation.turns) == 2
