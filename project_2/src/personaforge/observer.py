"""
observer.py — a third agent that watches the conversation and judges it.

Two layers, because they catch different things:

  RULE CHECKS (free, deterministic)
      Things a regex can see with certainty: "as an AI", asterisk stage
      directions, a line that speaks for another participant ("Maria: ..."),
      and verbatim repetition. These run on every conversation, cost nothing,
      and are what the tests pin down.

  MODEL JUDGE (one call per conversation)
      Things only a reader can see: did the conversation achieve its goal, did
      someone drift out of character in tone rather than words, what happened.
      Its answer is asked for as JSON and parsed generously — a judge that
      returns prose still produces a usable summary.

The summary it writes is also what each participant remembers afterwards
(memory.py), so the observer costs one call and pays for two features.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .conversation import Conversation
from .llm import LLM
from .persona import PersonaSpec

_OBSERVER_SYSTEM = """You are the observer of a role-play conversation between characters played by an AI.
You did not take part. Judge it honestly and briefly.

Return ONLY a JSON object with these keys:
{
  "summary": "two or three sentences on what actually happened",
  "goal_met": <integer 1-5, how well the conversation achieved the stated goal>,
  "goal_reason": "one sentence justifying that score",
  "in_character": {"<participant name>": true or false, ...},
  "flags": ["short notes on real problems: broke character, repeated itself, lost the thread, spoke for someone else, never answered the question"]
}
Use an empty list for flags if there were no problems. No text outside the JSON."""


@dataclass
class Evaluation:
    """The observer's verdict on one conversation."""

    summary: str
    goal: str
    goal_met: Optional[int] = None  # 1–5, or None if the judge did not give one
    goal_reason: str = ""
    in_character: Dict[str, bool] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)  # model judge
    rule_flags: List[str] = field(default_factory=list)  # deterministic checks

    @property
    def all_flags(self) -> List[str]:
        return self.rule_flags + self.flags

    def to_markdown(self) -> str:
        """Render as a section appended to the saved transcript."""
        score = f"{self.goal_met}/5" if self.goal_met is not None else "not scored"
        lines = [
            "## Observer",
            "",
            f"**Summary:** {self.summary}",
            "",
            f"**Goal:** {self.goal} — **{score}**" + (f". {self.goal_reason}" if self.goal_reason else ""),
        ]
        if self.in_character:
            lines += ["", "**In character:** " + ", ".join(
                f"{name} {'✓' if ok else '✗'}" for name, ok in self.in_character.items()
            )]
        lines += ["", "**Flags:**"] + ([f"- {f}" for f in self.all_flags] or ["- none"])
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Layer 1: rule checks
# -----------------------------------------------------------------------------

_AI_TELLS = re.compile(r"\b(?:as an ai|language model|i'?m an ai|i am an ai|this simulation|role-?play)\b", re.I)


def rule_flags(conversation: Conversation) -> List[str]:
    """Problems that can be detected with certainty, without a model call."""
    flags: List[str] = []
    others = {p.lower() for p in conversation.participants}
    seen: Dict[str, str] = {}

    for i, turn in enumerate(conversation.turns, start=1):
        if _AI_TELLS.search(turn.text):
            flags.append(f"line {i}: {turn.speaker} referred to being an AI or a simulation")
        if re.search(r"\*[^*\n]+\*", turn.text):
            flags.append(f"line {i}: {turn.speaker} used an *asterisk* stage direction")
        for line in turn.text.splitlines():
            prefix = line.split(":", 1)[0].strip().strip("*").lower() if ":" in line else ""
            if prefix in others and prefix != turn.speaker.lower():
                flags.append(f"line {i}: {turn.speaker} wrote a line for {line.split(':', 1)[0].strip('* ')}")
        key = " ".join(turn.text.lower().split())
        if key in seen:
            flags.append(f"line {i}: {turn.speaker} repeated an earlier line word for word")
        seen[key] = turn.speaker

    if conversation.error:
        flags.append(f"conversation cut short: {conversation.error}")
    return flags


# -----------------------------------------------------------------------------
# Layer 2: model judge
# -----------------------------------------------------------------------------


def _extract_json(raw: str) -> Optional[dict]:
    """Find the first {...} object in the reply, tolerating fences and preamble."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_evaluation(raw: str, goal: str, participants: List[str]) -> Evaluation:
    """
    Turn the judge's reply into an Evaluation, whatever shape it came back in.

    Strict about types (a score must be an int from 1 to 5), generous about
    everything else (missing keys, prose instead of JSON).
    """
    data = _extract_json(raw)
    if data is None:
        # The judge ignored the format. Its prose is still a summary.
        return Evaluation(summary=raw.strip()[:500] or "(no summary)", goal=goal)

    score = data.get("goal_met")
    try:
        score = int(score)
        score = score if 1 <= score <= 5 else None
    except (TypeError, ValueError):
        score = None

    raw_ic = data.get("in_character")
    in_character = (
        {str(k): bool(v) for k, v in raw_ic.items() if str(k) in participants}
        if isinstance(raw_ic, dict)
        else {}
    )
    flags = data.get("flags")
    return Evaluation(
        summary=str(data.get("summary") or "(no summary)").strip(),
        goal=goal,
        goal_met=score,
        goal_reason=str(data.get("goal_reason") or "").strip(),
        in_character=in_character,
        flags=[str(f) for f in flags] if isinstance(flags, list) else [],
    )


def evaluate(
    conversation: Conversation,
    llm: LLM,
    goal: Optional[str] = None,
    personas: Optional[List[PersonaSpec]] = None,
) -> Evaluation:
    """
    Judge a finished conversation: rule checks, then one model call.

    :param goal:     what the conversation was supposed to achieve. Defaults to
                     making real progress on the topic.
    :param personas: if given, each character's summary is shown to the judge
                     so it can tell whether they stayed in character.
    """
    goal = goal or f"make real progress on: {conversation.topic}"
    cast = "\n".join(f"- {p.name} ({p.role}): {p.summary}" for p in personas or [])
    prompt = (
        f"Goal: {goal}\n\n"
        + (f"Characters:\n{cast}\n\n" if cast else "")
        + f"Transcript:\n\n{conversation.transcript()}"
    )

    if conversation.turns:
        raw = llm.complete(_OBSERVER_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=800)
        evaluation = parse_evaluation(raw, goal, conversation.participants)
    else:
        # Nothing was said; there is nothing to judge and no reason to pay for it.
        evaluation = Evaluation(summary="Nobody spoke.", goal=goal)

    evaluation.rule_flags = rule_flags(conversation)
    return evaluation
