"""
Tests for the persona.py YOUR TASK items: overwrite protection, disambiguation,
quoting, validation with one retry, and in-app editing.
"""

import pytest

from personaforge import (
    AmbiguousPersonaError,
    PersonaNotFoundError,
    PersonaSpec,
    ScriptedLLM,
    edit_persona,
    find_persona,
    generate_persona,
    require_persona,
    save_persona,
)
from personaforge.persona import normalize_role, validation_problems

GOOD = """---
name: Ruth Okafor
role: Emergency Physician
summary: ER doctor: twelve years, no patience left
---

# Ruth Okafor

## Background
Twelve years in the ER.

## Personality
- Dry

## How they speak
- Short
"""


def save(tmp_path, name, role="x"):
    return save_persona(PersonaSpec(name=name, role=role, summary="", body="b"), tmp_path)


# -- saving -------------------------------------------------------------------


def test_save_never_overwrites_by_default(tmp_path):
    first = save(tmp_path, "Maria Delgado")
    second = save(tmp_path, "Maria Delgado")
    assert (first.name, second.name) == ("maria-delgado.md", "maria-delgado-2.md")


def test_save_with_overwrite_edits_in_place(tmp_path):
    path = save(tmp_path, "Maria Delgado")
    spec = require_persona("maria", tmp_path)
    spec.summary = "changed"
    assert save_persona(spec, tmp_path, overwrite=True) == path
    assert len(list(tmp_path.glob("*.md"))) == 1


# -- finding ------------------------------------------------------------------


def test_prefix_does_not_match_inside_a_word(tmp_path):
    """The skeleton's substring match let 'doc' find 'Dorothy'... and 'ann' find 'Joanna'."""
    save(tmp_path, "Joanna Ruiz", role="teacher")
    assert find_persona("ann", tmp_path) is None


def test_role_prefix_matches(tmp_path):
    save(tmp_path, "Dorothy Hale", role="doctor")
    assert find_persona("doc", tmp_path).name == "Dorothy Hale"


def test_titles_and_articles_are_ignored(tmp_path):
    save(tmp_path, "Dr. Samuel Reyes", role="doctor")
    assert find_persona("the doctor", tmp_path).name == "Dr. Samuel Reyes"
    assert find_persona("Dr Reyes", tmp_path).name == "Dr. Samuel Reyes"


def test_same_role_is_ambiguous(tmp_path):
    save(tmp_path, "Ann Lee", role="doctor")
    save(tmp_path, "Raj Patel", role="doctor")
    with pytest.raises(AmbiguousPersonaError) as info:
        find_persona("doctor", tmp_path)
    assert "Ann Lee" in str(info.value) and "Raj Patel" in str(info.value)
    assert find_persona("raj", tmp_path).name == "Raj Patel"


def test_duplicate_names_are_reachable_by_file_name(tmp_path):
    save(tmp_path, "Maria Delgado", role="patient")
    save(tmp_path, "Maria Delgado", role="nurse")
    with pytest.raises(AmbiguousPersonaError):
        find_persona("maria", tmp_path)
    assert find_persona("maria-delgado-2", tmp_path).role == "nurse"


def test_exact_name_beats_prefix(tmp_path):
    save(tmp_path, "Ann Lee")
    save(tmp_path, "Ann Leeson")
    assert find_persona("Ann Lee", tmp_path).name == "Ann Lee"


def test_require_persona_lists_options(tmp_path):
    save(tmp_path, "Ann Lee")
    with pytest.raises(PersonaNotFoundError) as info:
        require_persona("zaphod", tmp_path)
    assert "Ann Lee" in str(info.value)


def test_blank_name_finds_nothing(tmp_path):
    save(tmp_path, "Ann Lee")
    assert find_persona("the", tmp_path) is None


# -- parsing ------------------------------------------------------------------


def test_colon_inside_a_value_survives():
    assert PersonaSpec.from_markdown(GOOD).summary == "ER doctor: twelve years, no patience left"


def test_quoted_values_are_unquoted():
    text = '---\nname: "Ruth Okafor"\nrole: \'doctor\'\nsummary: "says \\"no\\" a lot"\n---\nbody'
    spec = PersonaSpec.from_markdown(text)
    assert (spec.name, spec.role, spec.summary) == ("Ruth Okafor", "doctor", 'says "no" a lot')


def test_values_that_need_quotes_roundtrip():
    spec = PersonaSpec(name="Ruth", role="doctor", summary='"Quoted" start', body="b")
    assert PersonaSpec.from_markdown(spec.to_markdown()).summary == '"Quoted" start'


# -- validation ---------------------------------------------------------------


def test_normalize_role():
    assert normalize_role("Emergency Physician") == "physician"
    assert normalize_role("ER nurse!") == "nurse"
    assert normalize_role("???") == "person"


def test_good_persona_has_no_problems():
    assert validation_problems(PersonaSpec.from_markdown(GOOD)) == []


def test_prose_body_and_bad_name_are_problems():
    problems = validation_problems(PersonaSpec(name="Patient 42", role="x", summary="", body="Sure! Here you go."))
    assert len(problems) == 2


def test_generation_retries_once_and_uses_the_fix():
    llm = ScriptedLLM(replies=["Sure! I'd love to help.", GOOD])
    spec = generate_persona("an ER doctor", llm)

    assert spec.name == "Ruth Okafor"
    assert spec.role == "physician"  # normalised to one word
    assert len(llm.calls) == 2
    assert "does not follow the required structure" in llm.calls[1]["messages"][-1]["content"]


def test_generation_keeps_first_answer_if_retry_is_no_better():
    llm = ScriptedLLM(replies=[GOOD.replace("## Personality", "## Mood").replace("## How they speak", "## Voice"), "nope"])
    assert generate_persona("an ER doctor", llm).name == "Ruth Okafor"


# -- editing ------------------------------------------------------------------


def test_edit_keeps_name_and_path(tmp_path):
    original = PersonaSpec("Ann Lee", "doctor", "s", "b")
    save_persona(original, tmp_path)
    revised = edit_persona(original, "rename her", ScriptedLLM(replies=[GOOD]))

    assert revised.name == "Ann Lee"
    assert revised.path == original.path
    assert "Twelve years in the ER" in revised.body


def test_malformed_edit_returns_the_original():
    original = PersonaSpec("Ann Lee", "doctor", "s", "b")
    assert edit_persona(original, "x", ScriptedLLM(replies=["no"])) is original
