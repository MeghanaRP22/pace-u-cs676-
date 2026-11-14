"""
Tests for the observer (rule checks + model judge) and per-persona memory.
"""

from datetime import date

from personaforge import Conversation, MemoryStore, PersonaSpec, ScriptedLLM, Turn, agent_from_spec, evaluate
from personaforge.observer import parse_evaluation, rule_flags

PEOPLE = ["Ann Lee", "Bo Kim"]


def convo(*lines, error=None):
    c = Conversation(topic="the scan", participants=PEOPLE, turns=[Turn(s, t) for s, t in lines])
    c.error = error
    return c


# -- rule checks --------------------------------------------------------------


def test_clean_conversation_has_no_rule_flags():
    assert rule_flags(convo(("Ann Lee", "How are you?"), ("Bo Kim", "Tired."))) == []


def test_ai_tell_is_flagged():
    flags = rule_flags(convo(("Ann Lee", "As an AI, I cannot diagnose you.")))
    assert len(flags) == 1 and "AI" in flags[0]


def test_asterisk_stage_direction_is_flagged():
    assert "asterisk" in rule_flags(convo(("Bo Kim", "*sighs heavily* Fine.")))[0]


def test_speaking_for_someone_else_is_flagged():
    flags = rule_flags(convo(("Ann Lee", "Let's begin.\nBo Kim: Okay doctor.")))
    assert "wrote a line for Bo Kim" in flags[0]


def test_verbatim_repetition_is_flagged():
    flags = rule_flags(convo(("Ann Lee", "I see."), ("Bo Kim", "Yes."), ("Ann Lee", "I  see.")))
    assert "repeated" in flags[0]


def test_error_is_flagged():
    assert "cut short" in rule_flags(convo(("Ann Lee", "hi"), error="Bo Kim: LLMError: down"))[0]


# -- parsing the judge --------------------------------------------------------


def test_parse_clean_json():
    ev = parse_evaluation(
        '{"summary": "S", "goal_met": 4, "goal_reason": "R", "in_character": {"Ann Lee": true, "Bo Kim": false}, "flags": ["f"]}',
        "g",
        PEOPLE,
    )
    assert (ev.summary, ev.goal_met, ev.goal_reason, ev.flags) == ("S", 4, "R", ["f"])
    assert ev.in_character == {"Ann Lee": True, "Bo Kim": False}


def test_parse_json_inside_a_code_fence_with_preamble():
    ev = parse_evaluation('Here you go:\n```json\n{"summary": "S", "goal_met": "5"}\n```', "g", PEOPLE)
    assert ev.summary == "S" and ev.goal_met == 5


def test_out_of_range_or_bad_score_becomes_none():
    assert parse_evaluation('{"summary": "S", "goal_met": 9}', "g", PEOPLE).goal_met is None
    assert parse_evaluation('{"summary": "S", "goal_met": "high"}', "g", PEOPLE).goal_met is None


def test_unknown_names_in_in_character_are_dropped():
    ev = parse_evaluation('{"summary": "S", "in_character": {"Zaphod": true}}', "g", PEOPLE)
    assert ev.in_character == {}


def test_prose_reply_still_yields_a_summary():
    ev = parse_evaluation("They talked about the scan and agreed on nothing.", "g", PEOPLE)
    assert ev.summary.startswith("They talked") and ev.goal_met is None


def test_broken_json_falls_back_to_prose():
    assert parse_evaluation("{not json at all}", "g", PEOPLE).summary == "{not json at all}"


# -- evaluate() ---------------------------------------------------------------


def test_evaluate_sends_goal_cast_and_transcript():
    llm = ScriptedLLM(replies=['{"summary": "S", "goal_met": 3}'])
    personas = [PersonaSpec("Ann Lee", "doctor", "blunt internist", "b")]
    ev = evaluate(convo(("Ann Lee", "UNIQUE_LINE")), llm, goal="explain the scan", personas=personas)

    sent = llm.calls[0]["messages"][0]["content"]
    assert "explain the scan" in sent and "blunt internist" in sent and "UNIQUE_LINE" in sent
    assert ev.goal_met == 3


def test_evaluate_skips_the_model_when_nobody_spoke():
    llm = ScriptedLLM()
    ev = evaluate(convo(), llm)
    assert llm.calls == [] and ev.summary == "Nobody spoke."


def test_evaluation_markdown_lists_flags():
    llm = ScriptedLLM(replies=['{"summary": "S", "goal_met": 2, "flags": ["drifted"]}'])
    md = evaluate(convo(("Ann Lee", "*waves*")), llm).to_markdown()
    assert "2/5" in md and "drifted" in md and "asterisk" in md


# -- memory -------------------------------------------------------------------


def spec(tmp_path, name="Ann Lee"):
    s = PersonaSpec(name=name, role="doctor", summary="", body="b")
    s.path = tmp_path / f"{s.slug()}.md"
    return s


def test_remember_and_recall(tmp_path):
    store, ann = MemoryStore(tmp_path), spec(tmp_path)
    store.remember(ann, "Met Bo.\nHe was tired.", when=date(2026, 1, 2))

    assert store.entries(ann) == ["2026-01-02: Met Bo. He was tired."]
    assert store.recall(ann) == "- 2026-01-02: Met Bo. He was tired."
    assert store.path_for(ann) == tmp_path / "memory" / "ann-lee.md"


def test_recall_is_bounded(tmp_path):
    store, ann = MemoryStore(tmp_path, max_entries=3), spec(tmp_path)
    for i in range(10):
        store.remember(ann, f"event {i}")
    recalled = store.recall(ann).splitlines()
    assert len(recalled) == 3 and recalled[-1].endswith("event 9")


def test_memories_are_private(tmp_path):
    store = MemoryStore(tmp_path)
    store.remember(spec(tmp_path, "Ann Lee"), "ANN_SECRET")
    assert store.recall(spec(tmp_path, "Bo Kim")) == ""


def test_forget(tmp_path):
    store, ann = MemoryStore(tmp_path), spec(tmp_path)
    store.remember(ann, "x")
    assert store.forget(ann) is True
    assert store.forget(ann) is False and store.entries(ann) == []


def test_memory_reaches_the_system_prompt_only_when_present():
    s = PersonaSpec("Ann Lee", "doctor", "", "b")
    assert "REMEMBER" not in agent_from_spec(s, ScriptedLLM()).system_prompt()
    prompt = agent_from_spec(s, ScriptedLLM(), memory="- saw Bo about his back").system_prompt()
    assert "saw Bo about his back" in prompt and "private" in prompt
