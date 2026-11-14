"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through one of two
protocols below:

    LLM      complete(system, messages) -> str
             All a persona ever needs: who you are, what was said, your line.

    ToolLLM  complete_with_tools(system, messages, tools) -> ToolTurn
             What the orchestrator needs: the model may answer in text, or ask
             for one or more tools to be run.

Implementations:

    ClaudeLLM        real Claude calls, both protocols. Needs ANTHROPIC_API_KEY.
    ScriptedLLM      canned strings, LLM only. No key, no network, no cost.
    ScriptedToolLLM  canned ToolTurns, for testing the orchestrator's tool loop.
    DemoLLM          the `--offline` stub, both protocols.

That split is the reason this project is testable: every test injects a
scripted model, so the suite runs in well under a second for free.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from .errors import LLMError

# Claude Opus 5 is the most capable model. While you are iterating you may want
# "claude-haiku-4-5" — a persona conversation makes one call per turn, and those
# add up. Say which model produced your submitted transcripts.
DEFAULT_MODEL = "claude-opus-5"

Message = Dict[str, Any]


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        ...


@dataclass
class ToolCall:
    """One tool the model asked us to run."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolTurn:
    """
    One model response in a tool-use loop.

    `content` is the response in Anthropic's block format, ready to be appended
    to the history as the assistant message. The API requires that exact echo
    before the matching tool_result blocks, so we keep it rather than rebuild it.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    content: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def say(cls, text: str) -> "ToolTurn":
        """A plain text reply — the model is done, or is asking the user something."""
        return cls(text=text, content=[{"type": "text", "text": text}])

    @classmethod
    def use(cls, *calls: "tuple[str, Dict[str, Any]]", text: str = "") -> "ToolTurn":
        """A reply that requests tools: ToolTurn.use(("list_personas", {}), ...)."""
        turn = cls(text=text)
        if text:
            turn.content.append({"type": "text", "text": text})
        for i, (name, args) in enumerate(calls):
            call = ToolCall(id=f"toolu_{_next_id()}_{i}", name=name, input=dict(args))
            turn.tool_calls.append(call)
            turn.content.append({"type": "tool_use", "id": call.id, "name": name, "input": call.input})
        return turn


_counter = [0]


def _next_id() -> int:
    _counter[0] += 1
    return _counter[0]


@runtime_checkable
class ToolLLM(Protocol):
    """An LLM that can also choose tools."""

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        ...

    def complete_with_tools(
        self, system: str, messages: List[Message], tools: List[Dict[str, Any]], max_tokens: int = 2048
    ) -> ToolTurn:
        ...


def supports_tools(llm: Any) -> bool:
    """True if this model can drive the tool-use orchestrator."""
    return callable(getattr(llm, "complete_with_tools", None))


# =============================================================================
# ClaudeLLM — needs your own API key; calls are billed to you
# =============================================================================


@dataclass
class ClaudeLLM:
    """
    Real Claude.

    The client is created lazily so importing this module never requires a key.
    Every call is counted in `usage` so the app can tell you what a
    conversation actually cost, which the failure analysis needs.
    """

    model: str = DEFAULT_MODEL
    usage: Dict[str, int] = field(
        default_factory=lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            # The SDK already retries connection errors, 429s and 5xx responses
            # with backoff; three attempts is enough for a classroom demo.
            self._client = anthropic.Anthropic(max_retries=3)
        return self._client

    def _create(self, **kwargs: Any) -> Any:
        """One API request, with SDK errors translated into an actionable LLMError."""
        import anthropic

        try:
            response = self._get_client().messages.create(model=self.model, **kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Your API key was rejected. Check ANTHROPIC_API_KEY in .env.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model '{self.model}' was not found. Try --model claude-haiku-4-5.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the API. Wait a minute and try again.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API. Check your network connection.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The API returned an error ({exc.status_code}): {exc.message}") from exc

        self.usage["calls"] += 1
        if getattr(response, "usage", None) is not None:
            self.usage["input_tokens"] += response.usage.input_tokens
            self.usage["output_tokens"] += response.usage.output_tokens
        return response

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        response = self._create(max_tokens=max_tokens, system=system, messages=messages)

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return "[declined to respond]"
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def complete_with_tools(
        self, system: str, messages: List[Message], tools: List[Dict[str, Any]], max_tokens: int = 2048
    ) -> ToolTurn:
        """Send one request with tools available and return what the model chose."""
        response = self._create(max_tokens=max_tokens, system=system, messages=messages, tools=tools)

        turn = ToolTurn()
        for block in response.content:
            # Rebuild plain dicts by hand: echoing SDK objects back can carry
            # fields (citations=None, ...) that some API versions reject.
            if block.type == "text":
                turn.text += block.text
                turn.content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                turn.tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
                turn.content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": dict(block.input)}
                )
        turn.text = turn.text.strip()
        if response.stop_reason == "refusal" and not turn.tool_calls:
            return ToolTurn.say("[declined to respond]")
        return turn


# =============================================================================
# Scripted models — for tests
# =============================================================================


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    It records every call in `.calls`, which makes it easy to assert that a
    persona's markdown actually reached the system prompt. It does NOT support
    tools, so an Orchestrator given one falls back to pattern routing.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


@dataclass
class ScriptedToolLLM:
    """
    A fake tool-using model.

    `tool_turns` is replayed in order by complete_with_tools() — the orchestrator
    side. `replies` serves plain complete() calls — personas, the observer, and
    the persona author. Keeping the two queues separate means a test can script
    "the model calls create_persona twice" without caring how many dialogue
    calls happen in between.
    """

    tool_turns: List[ToolTurn] = field(default_factory=list)
    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_seen: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]

    def complete_with_tools(
        self, system: str, messages: List[Message], tools: List[Dict[str, Any]], max_tokens: int = 2048
    ) -> ToolTurn:
        self.tool_calls_seen.append({"system": system, "messages": list(messages), "tools": tools})
        index = len(self.tool_calls_seen) - 1
        if index < len(self.tool_turns):
            return self.tool_turns[index]
        return ToolTurn.say("(done)")


# =============================================================================
# DemoLLM — the `--offline` stub
# =============================================================================

_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again. [END]",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""

_DEMO_EVALUATION = {
    "summary": "Offline demo: a scripted exchange about symptoms that ends with tests being ordered.",
    "goal_met": 3,
    "goal_reason": "The scripted lines move toward a plan but never discuss results.",
    "flags": ["offline mode — scripted dialogue, not a real judgement"],
}


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    It looks at the system prompt and answers in the shape the caller expects:
    persona markdown for the author prompt, JSON for the observer, dialogue for
    personas. For tool use it stands in for Claude's choice of tools using the
    pattern planner in router.py — so the full tool loop runs offline, but the
    *understanding* is only as good as those patterns. Use a real key to see
    the real thing.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)
    _dialogue_index: int = 0

    def complete(self, system: str, messages: List[Message], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = str(request).replace("Write a persona for:", "").strip()
            if description.startswith("Here is an existing persona"):
                # An edit: return the original document unchanged.
                return description.split("\n\n", 1)[1].rsplit("\n\nRevise it:", 1)[0]
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        if "stage manager" in system.lower():
            return "Offline mode — I'm a stub. Try: create a persona doctor"

        if system.startswith("You are the observer"):
            return json.dumps(_DEMO_EVALUATION)

        line = _DEMO_DIALOGUE[self._dialogue_index % len(_DEMO_DIALOGUE)]
        self._dialogue_index += 1
        return line

    def complete_with_tools(
        self, system: str, messages: List[Message], tools: List[Dict[str, Any]], max_tokens: int = 2048
    ) -> ToolTurn:
        from .router import plan  # local import: router imports nothing from here

        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        last = messages[-1]["content"] if messages else ""

        # After tools ran, a real model would summarize. Echo the first line of
        # each result, which is what the orchestrator wrote for the human anyway.
        if isinstance(last, list):
            lines = [str(block.get("content", "")).splitlines()[0] for block in last if block.get("content")]
            return ToolTurn.say("\n\n".join(lines) or "Done.")

        steps = plan(str(last))
        if not steps:
            return ToolTurn.say("Offline mode — I'm a stub. Try: create a persona doctor")
        return ToolTurn.use(*steps)


def default_llm(offline: bool = False, model: Optional[str] = None) -> Union[ClaudeLLM, DemoLLM]:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
