"""
router.py — the pattern planner, demoted to a fallback.

This is what the skeleton's regex router became. It is no longer how the app
understands you when a real model is available: Orchestrator hands Claude a
set of tools and lets it choose. This module survives for two jobs only:

  1. Fallback routing for a model that cannot call tools (ScriptedLLM in the
     tests, or any LLM you plug in that only implements `complete`).
  2. Standing in for Claude's tool choice inside DemoLLM, so `--offline`
     exercises the real tool loop end to end without a key.

It returns the same thing Claude returns — a list of (tool_name, arguments) —
so both paths go through the one tool executor in orchestrator.py. That shared
shape is the point: there is exactly one place where tools actually run.

Improvements over the skeleton, so the offline demo is not embarrassing:
  * "then" / ";" split a request into ordered steps, so "list the personas
    then have them talk" does both instead of only listing.
  * Counts: "two doctors and a nurse" plans three create_persona calls.
  * "N turns" is honoured.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

Step = Tuple[str, Dict[str, Any]]

_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# Clause boundaries: "... then ...", "...; ...", "... and have them ...".
_CLAUSE_SPLIT = re.compile(
    r"\s*;\s*|\s*,?\s+(?:and\s+)?then\s+|\s+and\s+(?=(?:have|let|get|start|make them)\b)"
)


def _singular(phrase: str) -> str:
    """'doctors who argue' -> 'doctor who argue'. Only the head noun matters here."""
    head, _, tail = phrase.partition(" ")
    if head.endswith("ies"):
        head = head[:-3] + "y"
    elif head.endswith("s") and not head.endswith("ss"):
        head = head[:-1]
    return f"{head} {tail}".strip()


def _create_steps(description: str) -> List[Step]:
    """Expand 'two doctors and a nurse' into one create_persona step per person."""
    description = re.sub(r"^(?:(?:a|an|new|another)\s+)?personas?\s*(?:for|of|who is|that is|:)?\s*", "", description)
    # Only split on "and"/"," when a new count follows, so "a doctor who is
    # kind and patient" stays one persona.
    pieces = re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:a|an|one|two|three|four|five|\d+)\b)", description)
    steps: List[Step] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        match = re.match(r"^(a|an|one|two|three|four|five|\d+)\s+(.*)$", piece)
        count, rest = 1, piece
        if match:
            word, rest = match.groups()
            count = int(word) if word.isdigit() else _NUMBERS[word]
            if count > 1:
                rest = _singular(rest)
        steps += [("create_persona", {"description": rest})] * min(count, 5)
    if not steps:
        # "create a persona" with no description: let the executor ask.
        steps.append(("create_persona", {"description": ""}))
    return steps


def _conversation_step(clause: str) -> Step:
    """Pull participants, topic, and turn count out of a 'talk about' clause."""
    args: Dict[str, Any] = {}

    topic = re.search(r"\b(?:about|over|on)\s+(.*?)(?:\s+for\s+\d+\s+turns?)?$", clause)
    args["topic"] = topic.group(1).strip() if topic else "an initial consultation"

    who = re.search(r"\bbetween\s+(.*?)(?:\s+(?:about|over|on)\b|$)", clause)
    if who:
        args["participants"] = [w.strip() for w in re.split(r"\s+and\s+|,", who.group(1)) if w.strip()]

    turns = re.search(r"\b(\d+)\s+turns?\b", clause)
    if turns:
        args["max_turns"] = int(turns.group(1))
    return ("start_conversation", args)


def plan(text: str) -> List[Step]:
    """
    Turn free text into an ordered list of tool calls, or [] if nothing matched.

    An empty plan means "this was conversation, not a command" — the caller
    should just talk back.
    """
    steps: List[Step] = []
    for clause in _CLAUSE_SPLIT.split(text.strip().lower()):
        clause = clause.strip(" .!?")
        if not clause:
            continue

        # Order matters only between patterns that share a verb ("make"), and
        # each is anchored tightly enough that it cannot steal another's input.
        edit = re.search(r"\b(?:make|edit|change|update)\s+(.+?)\s+((?:more|less)\s+.+)$", clause)
        show = re.search(r"\b(?:show|open|read)\s+(?:me\s+)?(?:the\s+)?persona\s+(?:for\s+|of\s+)?(.+)$", clause)
        create = re.search(r"\b(?:create|make|add|build|generate)\s+(?:it\s+|me\s+|us\s+)?(.*)$", clause)

        if edit and not re.match(r"(?:a|an|new|another)\s", edit.group(1)):
            steps.append(("edit_persona", {"name": edit.group(1), "instruction": edit.group(2)}))
        elif show:
            steps.append(("show_persona", {"name": show.group(1)}))
        elif re.search(r"\b(?:talk|conversation|converse|chat|discuss|speak|argue|debate)\b", clause):
            steps.append(_conversation_step(clause))
        elif create and re.search(r"\bpersonas?\b|^(?:a|an|one|two|three|four|five|\d+)\s", create.group(1)):
            steps += _create_steps(create.group(1))
        elif re.search(r"\b(?:list|show|who)\b.*\bpersonas?\b", clause) or clause in {"list", "ls"}:
            steps.append(("list_personas", {}))
    return steps
