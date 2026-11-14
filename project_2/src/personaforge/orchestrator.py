"""
orchestrator.py — the agent you talk to.

You type "make two doctors and a nurse and have them argue about the
diagnosis". Something has to turn that into four actions in the right order.

The skeleton did this with a stack of regular expressions. This version does
what a modern agent does: it hands Claude a set of tools, lets the model decide
which to call (possibly several, possibly in sequence), runs them, and sends
the results back until the model has nothing left to do. That buys three
things the regexes could never do:

  * Any phrasing works — the model reads intent, not keywords.
  * It can ask a clarifying question instead of guessing ("what kind of
    doctor?"), because "reply in text" is always one of its options.
  * Multi-step requests just work: the loop keeps going while the model keeps
    asking for tools.

If the LLM cannot call tools at all (a ScriptedLLM in tests, or a plug-in model
with only `complete`), the orchestrator falls back to router.plan(), which
produces the same (tool, arguments) steps from patterns. Both paths run through
the single executor below, so a tool behaves identically however it was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .agent import agent_from_spec
from .conversation import DEFAULT_TURNS, Conversation, Turn, run_conversation, save_transcript
from .errors import PersonaForgeError
from .llm import LLM, supports_tools
from .memory import MemoryStore, memory_entry
from .observer import Evaluation, evaluate
from .persona import (
    DEFAULT_PERSONA_DIR,
    PersonaSpec,
    edit_persona,
    generate_persona,
    list_personas,
    require_persona,
    save_persona,
)
from .router import plan

MAX_TURNS_LIMIT = 20  # hard ceiling on one conversation, whatever the model asks for
MAX_TOOL_ROUNDS = 8  # tool-use round trips per user message, to stop runaway loops
HISTORY_LIMIT = 30  # messages of orchestrator history kept between user messages

_STAGE_MANAGER_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters ("personas") and has them talk to each other.
You have tools to create, list, show, and edit personas, and to start a
conversation between them. Use them.

How to behave:
- When the user asks for something your tools can do, call the tools. You may call
  several in one turn — e.g. three create_persona calls for "two doctors and a nurse".
- If a request is too vague to act on (for example "create a persona" with no hint of
  who), ask ONE short clarifying question instead of inventing details.
- If the user says "have them talk" right after creating personas, the participants
  are the personas just created.
- After tools run, reply in one or two sentences. Do not repeat the dialogue, persona
  lists, or character sheets — the user already sees those. Mention the observer's
  score if there was one.
- Otherwise, be brief: two or three sentences."""

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "create_persona",
        "description": (
            "Create and save ONE new persona character from a short description. "
            "Call it once per character; call it several times for several characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Who to create, e.g. 'an ER nurse on her third night shift who is out of patience'.",
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "list_personas",
        "description": "List every saved persona with its role and one-line summary.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "show_persona",
        "description": "Show the full character sheet of one persona.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Name, file name, or role."}},
            "required": ["name"],
        },
    },
    {
        "name": "edit_persona",
        "description": "Revise an existing persona, e.g. make them more skeptical or give them a new backstory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Which persona: name, file name, or role."},
                "instruction": {"type": "string", "description": "What to change."},
            },
            "required": ["name", "instruction"],
        },
    },
    {
        "name": "start_conversation",
        "description": (
            "Have two or more saved personas talk to each other. They take turns, may stay "
            "silent, and may end the conversation early. An observer scores it afterwards."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "participants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names (or roles) of who takes part. The first one opens. "
                    "Omit only if exactly two personas exist or the user means the ones just created.",
                },
                "topic": {"type": "string", "description": "The situation or subject, e.g. 'the MRI results'."},
                "max_turns": {
                    "type": "integer",
                    "minimum": 2,
                    "maximum": MAX_TURNS_LIMIT,
                    "description": f"Most lines of dialogue. Default {DEFAULT_TURNS}.",
                },
                "goal": {
                    "type": "string",
                    "description": "What the conversation should achieve, for the observer to score.",
                },
            },
            "required": ["topic"],
        },
    },
]


@dataclass
class Result:
    """What the orchestrator did, in a form any UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None
    evaluation: Optional[Evaluation] = None
    # Markdown the user should see verbatim (a persona list, a character
    # sheet), shown before the reply. Never left to the model to paraphrase.
    details: List[str] = field(default_factory=list)


@dataclass
class _Outcome:
    """What one tool call produced: text for the model, side effects for the user."""

    text: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None
    is_error: bool = False
    display: bool = False  # show `text` to the user as-is, not just to the model


def _unique_names(specs: List[PersonaSpec]) -> List[PersonaSpec]:
    """
    Make every participant's name unique before the conversation starts.

    An agent recognises its own past lines by name, so two characters both
    called "Maria Delgado" would each think the other's lines were their own.
    Colliding names get their file name appended; the files are not touched.
    """
    counts: Dict[str, int] = {}
    for spec in specs:
        counts[spec.name] = counts.get(spec.name, 0) + 1
    return [
        replace(s, name=f"{s.name} ({s.file_id})", path=s.path) if counts[s.name] > 1 else s for s in specs
    ]


class Orchestrator:
    """
    Turns what the user typed into persona actions.

    :param llm:          the model. With tool support you get the real agent;
                         without it, pattern routing.
    :param persona_dir:  where persona files, memories, and transcripts live.
    :param use_observer: judge each conversation (one extra call each).
    :param use_memory:   give personas memories of earlier conversations.
    """

    def __init__(
        self,
        llm: LLM,
        persona_dir: Path = DEFAULT_PERSONA_DIR,
        use_observer: bool = True,
        use_memory: bool = True,
    ) -> None:
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.use_observer = use_observer
        self.memory = MemoryStore(self.persona_dir) if use_memory else None
        self.history: List[Dict[str, Any]] = []
        # Personas created during the current request, so "have them talk"
        # means the ones you just made rather than whichever sort first.
        self._recent: List[PersonaSpec] = []
        self._last_evaluation: Optional[Evaluation] = None

    # -- the tools ----------------------------------------------------------

    def _create_persona(self, description: str = "", **_: Any) -> _Outcome:
        if not description.strip():
            return _Outcome(
                "No description given. Ask the user what kind of person to create.", is_error=True
            )
        spec = generate_persona(description, self.llm)
        path = save_persona(spec, self.persona_dir)
        self._recent.append(spec)
        return _Outcome(f"Created {spec.name} ({spec.role}) — {spec.summary}", events=[f"wrote {path}"])

    def _list_personas(self, **_: Any) -> _Outcome:
        specs = list_personas(self.persona_dir)
        if not specs:
            return _Outcome("No personas yet. Try: create a persona patient")
        lines = [f"- **{s.name}** ({s.role}) — {s.summary}  `{s.file_id}`" for s in specs]
        return _Outcome(f"{len(specs)} persona(s) in `{self.persona_dir}`:\n" + "\n".join(lines), display=True)

    def _show_persona(self, name: str = "", **_: Any) -> _Outcome:
        spec = require_persona(name, self.persona_dir)
        remembered = self.memory.recall(spec) if self.memory else ""
        text = f"Character sheet for {spec.name} (`{spec.path}`):\n\n{spec.body}"
        if remembered:
            text += f"\n\n## Memories\n{remembered}"
        return _Outcome(text, display=True)

    def _edit_persona(self, name: str = "", instruction: str = "", **_: Any) -> _Outcome:
        if not instruction.strip():
            return _Outcome("No change described. Ask the user what to change.", is_error=True)
        spec = require_persona(name, self.persona_dir)
        revised = edit_persona(spec, instruction, self.llm)
        if revised is spec:
            return _Outcome(f"The revision of {spec.name} came back malformed; the file was left unchanged.", is_error=True)
        path = save_persona(revised, self.persona_dir, overwrite=True)
        return _Outcome(f"Updated {revised.name}: {instruction}", events=[f"wrote {path}"])

    def _pick_participants(self, names: Optional[List[str]]) -> List[PersonaSpec]:
        """Resolve who takes part, refusing to guess when the choice is ambiguous."""
        if names:
            specs: List[PersonaSpec] = []
            for name in names:
                spec = require_persona(name, self.persona_dir)
                if spec.file_id not in {s.file_id for s in specs}:
                    specs.append(spec)
            return specs
        if len(self._recent) >= 2:
            return list(self._recent)
        everyone = list_personas(self.persona_dir)
        if len(everyone) > 2:
            raise PersonaForgeError(
                "There are more than two personas, so say who should talk: "
                + ", ".join(s.name for s in everyone)
            )
        return everyone

    def _start_conversation(
        self,
        topic: str = "",
        participants: Optional[List[str]] = None,
        max_turns: int = DEFAULT_TURNS,
        goal: Optional[str] = None,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
        **_: Any,
    ) -> _Outcome:
        topic = topic.strip() or "an initial consultation"
        specs = self._pick_participants(participants)
        if len(specs) < 2:
            return _Outcome(
                "I need at least two personas before they can talk. Create another one first.", is_error=True
            )

        try:
            turns = max(2, min(int(max_turns), MAX_TURNS_LIMIT))
        except (TypeError, ValueError):
            turns = DEFAULT_TURNS

        agents = [
            agent_from_spec(s, self.llm, memory=self.memory.recall(s) if self.memory else "")
            for s in _unique_names(specs)
        ]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)

        events: List[str] = []
        if self.use_observer:
            # A failing judge must not cost the user a conversation they already watched.
            try:
                conversation.evaluation = evaluate(conversation, self.llm, goal=goal, personas=specs)
            except PersonaForgeError as exc:
                events.append(f"observer skipped: {exc}")
        self._last_evaluation = conversation.evaluation

        path = save_transcript(conversation, self.persona_dir)
        events.append(f"wrote {path}")

        if self.memory is not None and conversation.turns:
            summary = (
                conversation.evaluation.summary
                if conversation.evaluation is not None
                else f'It ended with {conversation.turns[-1].speaker} saying: "{conversation.turns[-1].text[:160]}"'
            )
            for spec in specs:
                self.memory.remember(spec, memory_entry(spec.name, conversation.participants, topic, summary))
            events.append(f"updated memories in {self.memory.directory}")

        text = (
            f"{len(conversation.turns)} lines between {', '.join(conversation.participants)}; "
            f"ended because {conversation.ended_by}."
        )
        if conversation.evaluation is not None:
            ev = conversation.evaluation
            score = f"{ev.goal_met}/5" if ev.goal_met is not None else "unscored"
            text += f"\nObserver: {ev.summary} Goal score {score}."
            if ev.all_flags:
                text += " Flags: " + "; ".join(ev.all_flags[:4])
        if conversation.error:
            text += f"\nWarning: {conversation.error}"
        return _Outcome(text, events=events, conversation=conversation, is_error=bool(conversation.error))

    # -- the single executor ------------------------------------------------

    def _run_tool(self, name: str, args: Dict[str, Any], **callbacks: Any) -> _Outcome:
        """
        Run one tool by name. Every route into a tool goes through here.

        Expected failures (unknown persona, ambiguity, API errors) become an
        error result the model can read and react to — usually by asking the
        user — instead of an exception that ends the turn.
        """
        handlers: Dict[str, Callable[..., _Outcome]] = {
            "create_persona": self._create_persona,
            "list_personas": self._list_personas,
            "show_persona": self._show_persona,
            "edit_persona": self._edit_persona,
            "start_conversation": self._start_conversation,
        }
        handler = handlers.get(name)
        if handler is None:
            return _Outcome(f"Unknown tool '{name}'.", is_error=True)
        if name == "start_conversation":
            args = {**args, **callbacks}
        try:
            return handler(**args)
        except PersonaForgeError as exc:
            return _Outcome(str(exc), is_error=True)

    # -- routing ------------------------------------------------------------

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """
        Work out what the user meant and do it.

        :param on_turn:  called with each line of dialogue as it is spoken.
        :param on_start: called with the topic just before a conversation starts.
        """
        self._recent = []
        self._last_evaluation = None
        callbacks = {"on_turn": on_turn, "on_start": on_start}
        if supports_tools(self.llm):
            return self._handle_with_tools(text, callbacks)
        return self._handle_with_patterns(text, callbacks)

    def _handle_with_tools(self, text: str, callbacks: Dict[str, Any]) -> Result:
        """The agent loop: call the model, run what it asks for, repeat."""
        self.history.append({"role": "user", "content": text})
        result = Result(reply="")

        for _ in range(MAX_TOOL_ROUNDS):
            turn = self.llm.complete_with_tools(_STAGE_MANAGER_SYSTEM, self.history, TOOLS, max_tokens=1024)
            self.history.append({"role": "assistant", "content": turn.content or [{"type": "text", "text": turn.text or "…"}]})

            if not turn.tool_calls:
                result.reply = turn.text or "Done."
                break

            tool_results = []
            for call in turn.tool_calls:
                outcome = self._run_tool(call.name, call.input, **callbacks)
                result.events += outcome.events
                if outcome.display:
                    result.details.append(outcome.text)
                if outcome.conversation is not None:
                    result.conversation = outcome.conversation
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": outcome.text,
                        **({"is_error": True} if outcome.is_error else {}),
                    }
                )
            self.history.append({"role": "user", "content": tool_results})
        else:
            result.reply = "I stopped after several steps without finishing. Try breaking the request up."

        result.evaluation = self._last_evaluation
        self._trim_history()
        return result

    def _handle_with_patterns(self, text: str, callbacks: Dict[str, Any]) -> Result:
        """Fallback for models without tool support: patterns choose, executor runs."""
        steps = plan(text)
        if not steps:
            return self._chat(text)

        result = Result(reply="")
        replies = []
        for name, args in steps:
            outcome = self._run_tool(name, args, **callbacks)
            if name == "create_persona" and outcome.is_error:
                # No model to ask the question for us, so ask it directly.
                outcome.text = "Tell me what kind of persona — for example, 'create a persona doctor'."
            replies.append(outcome.text)
            result.events += outcome.events
            if outcome.conversation is not None:
                result.conversation = outcome.conversation
        result.reply = "\n\n".join(replies)
        result.evaluation = self._last_evaluation
        return result

    def _chat(self, text: str) -> Result:
        """No command recognised and no tools available, so just talk."""
        self.history.append({"role": "user", "content": text})
        reply = self.llm.complete(_STAGE_MANAGER_SYSTEM, self.history, max_tokens=512)
        self.history.append({"role": "assistant", "content": reply})
        self._trim_history()
        return Result(reply=reply)

    def _trim_history(self) -> None:
        """
        Keep the orchestrator's memory bounded without breaking the API's rules.

        We may only cut at a plain user message (string content). Cutting
        anywhere else could leave a tool_result whose tool_use was dropped,
        which the API rejects.
        """
        if len(self.history) <= HISTORY_LIMIT:
            return
        for i in range(len(self.history) - HISTORY_LIMIT, len(self.history)):
            message = self.history[i]
            if message["role"] == "user" and isinstance(message["content"], str):
                self.history = self.history[i:]
                return

    def reset(self) -> None:
        """Forget the orchestrator's chat history. Personas and memories stay on disk."""
        self.history = []

