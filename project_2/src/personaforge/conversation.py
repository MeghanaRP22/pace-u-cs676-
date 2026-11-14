"""
conversation.py — put two or more agents in a room and let them talk.

The skeleton was strict round-robin for a fixed number of turns. This loop
keeps the same entry point but makes three decisions smarter:

  WHO SPEAKS NEXT   Whoever was just addressed by name or role. If nobody was,
                    whoever has been quiet longest. A rule, not a model call:
                    it is free, deterministic, testable, and with two people it
                    reduces to plain alternation.
  STAYING SILENT    An agent may reply [PASS]. If every participant passes in a
                    row, the room has gone quiet and the conversation ends.
  ENDING            An agent may close its line with [END] once everyone has
                    spoken at least once. `max_turns` is now a ceiling, not a
                    script.

And one robustness fix: if a model call fails mid-conversation, everything said
so far is kept, the failure is recorded, and the caller still gets a transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import END_TOKEN, PASS_TOKEN, PersonaAgent

DEFAULT_TURNS = 6

# How a conversation can finish. Kept as plain strings so they read well in a
# transcript and need no import to compare against.
ENDED_MAX_TURNS = "it reached the turn limit"
ENDED_BY_AGENT = "a participant closed it"
ENDED_SILENCE = "everyone fell silent"
ENDED_ERROR = "a model call failed"


@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)
    ended_by: str = ENDED_MAX_TURNS
    passes: Dict[str, int] = field(default_factory=dict)  # name -> times stayed silent
    error: Optional[str] = None
    evaluation: Any = None  # observer.Evaluation, attached after the fact

    def transcript(self) -> str:
        """Render as markdown, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        body = "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)
        footer = f"\n\n---\n\n_Ended: {self.ended_by} after {len(self.turns)} lines._"
        if self.passes:
            footer += " _Stayed silent: " + ", ".join(f"{k} ×{v}" for k, v in self.passes.items()) + "._"
        if self.error:
            footer += f"\n\n_Error: {self.error}_"
        if self.evaluation is not None:
            footer += "\n\n" + self.evaluation.to_markdown()
        return header + body + footer


def parse_control(text: str) -> "tuple[str, bool, bool]":
    """
    Split a raw agent reply into (dialogue, passed, wants_end).

    Generous on purpose: models write "[PASS]", "[pass].", or put [END] on its
    own line. Any of those count. A reply that is empty after stripping tokens
    is a pass, so a refusal or blank line never shows up as dialogue.
    """
    wants_end = bool(re.search(re.escape(END_TOKEN), text, re.IGNORECASE))
    tokens = f"{re.escape(PASS_TOKEN)}|{re.escape(END_TOKEN)}"
    clean = re.sub(tokens, "", text, flags=re.IGNORECASE).strip()
    passed = not clean or bool(re.fullmatch(r"\W*", clean))
    return ("" if passed else clean), passed, wants_end


_TITLES = {"dr", "mr", "mrs", "ms", "prof", "the"}


def _address_words(agent: PersonaAgent) -> List[str]:
    """Words that, said aloud, mean you are talking to this agent."""
    words = [w for w in re.findall(r"[a-z]+", agent.name.lower()) if w not in _TITLES and len(w) >= 2]
    return words + [agent.spec.role.lower()]


def next_speaker(
    agents: List[PersonaAgent], history: List[Turn], last: Optional[PersonaAgent], last_active: Dict[str, int]
) -> PersonaAgent:
    """
    Choose who speaks next.

    1. Nobody speaks twice in a row.
    2. If the last line names exactly one other participant (by first name,
       surname, or role), they answer.
    3. Otherwise, whoever has been silent longest; ties go to list order.
    """
    candidates = [a for a in agents if a is not last] or agents
    if last is not None and history:
        said = history[-1].text.lower()
        addressed = [
            a for a in candidates if any(re.search(rf"\b{re.escape(w)}\b", said) for w in _address_words(a))
        ]
        if len(addressed) == 1:
            return addressed[0]
    return min(candidates, key=lambda a: (last_active.get(a.name, -1), agents.index(a)))


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = DEFAULT_TURNS,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a conversation between the given agents.

    :param agents:  two or more PersonaAgents. The first one opens.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   the most lines of dialogue that will be spoken (passes do
                    not count). The conversation may end sooner.
    :param on_turn: called after each spoken line, so a UI can print as it goes.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")
    if turns < 1:
        raise ValueError("turns must be at least 1")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting line from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    history: List[Turn] = [Turn(speaker="Scene", text=topic)]
    last: Optional[PersonaAgent] = None
    last_active: Dict[str, int] = {}
    silent_streak = 0
    step = 0

    # Each pass costs a call but speaks no line, so cap total calls too.
    while len(conversation.turns) < turns and step < turns * 2:
        speaker = next_speaker(agents, history, last, last_active)
        last_active[speaker.name] = step
        step += 1

        try:
            raw = speaker.respond(history)
        except Exception as exc:  # noqa: BLE001 — keep what was said; report the failure
            conversation.error = f"{speaker.name}: {type(exc).__name__}: {exc}"
            conversation.ended_by = ENDED_ERROR
            return conversation

        text, passed, wants_end = parse_control(raw)
        last = speaker

        if passed:
            conversation.passes[speaker.name] = conversation.passes.get(speaker.name, 0) + 1
            silent_streak += 1
            if silent_streak >= len(agents):
                conversation.ended_by = ENDED_SILENCE
                return conversation
            continue

        silent_streak = 0
        turn = Turn(speaker=speaker.name, text=text)
        history.append(turn)
        conversation.turns.append(turn)
        if on_turn is not None:
            on_turn(turn)

        # An [END] is honoured only after everyone has had a chance to speak,
        # so an eager first speaker cannot close the room on its own.
        everyone_spoke = {t.speaker for t in conversation.turns} >= set(conversation.participants)
        if wants_end and everyone_spoke:
            conversation.ended_by = ENDED_BY_AGENT
            return conversation

    return conversation


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "conversation"


def save_transcript(conversation: Conversation, directory, filename: Optional[str] = None) -> Path:
    """
    Write the transcript under `<directory>/conversations/`.

    Default file names carry a timestamp and the topic, so a new conversation
    never overwrites an old one. Transcripts live in a subfolder so that
    list_personas(), which reads every `*.md` in the persona folder, never
    mistakes a transcript for a character.
    """
    folder = Path(directory) / "conversations"
    folder.mkdir(parents=True, exist_ok=True)
    name = filename or f"{datetime.now():%Y%m%d-%H%M%S}-{_slugify(conversation.topic)}.md"
    path = folder / name
    n = 2
    while filename is None and path.exists():
        path = folder / f"{Path(name).stem}-{n}.md"
        n += 1
    path.write_text(conversation.transcript(), encoding="utf-8")
    return path
