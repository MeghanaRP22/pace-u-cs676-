"""
agent.py — markdown file + model = an agent with its own mind.

A `PersonaAgent` holds a `PersonaSpec` (which came from a `.md` file), an
`LLM`, and optionally that persona's private memory. When you ask it to speak,
it builds a system prompt out of the markdown and sends the transcript so far.
The persona file *is* the agent's brain — nothing else distinguishes the
doctor from the patient.

Agents may also decline to speak ([PASS]) or close the conversation ([END]).
They signal this with tokens in their reply; conversation.py interprets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from .llm import LLM
from .persona import PersonaSpec

if TYPE_CHECKING:  # avoid a runtime import cycle with conversation.py
    from .conversation import Turn

PASS_TOKEN = "[PASS]"
END_TOKEN = "[END]"

# How many recent lines an agent sees. Every turn re-sends the transcript, so
# without a window a long conversation grows quadratically in cost and
# eventually overflows the context. The scene line is always kept.
DEFAULT_WINDOW = 20

# Wrapped around the persona markdown to put the model in character and keep it
# from narrating. Without the "speak only as" rule models tend to write stage
# directions and answer for everyone in the room.
_IN_CHARACTER_TEMPLATE = """You are role-playing a character in a simulated conversation.

Everything between the markers below defines who you are. Inhabit it completely:
speak in this person's voice, hold their opinions, want what they want.

=== BEGIN CHARACTER ===
{persona}
=== END CHARACTER ===
{memory}
Rules:
- Speak only as {name}. Never speak for anyone else.
- Reply with dialogue only. No stage directions, no asterisks, no narration.
- Stay in character even if the conversation goes somewhere unexpected.
- Keep replies to a few sentences unless the moment genuinely calls for more.
- You are {name}. Do not mention that you are an AI or a simulation.
- If you genuinely have nothing to add right now, reply with exactly {pass_token}.
- When the conversation has reached its natural end for you — your question is
  answered, a decision is made, or you are leaving — finish your last line with {end_token}."""

_MEMORY_BLOCK = """
=== WHAT YOU REMEMBER FROM EARLIER CONVERSATIONS ===
{entries}
=== END MEMORY ===
These memories are private to you. Bring them up only if they are relevant.
"""


@dataclass
class PersonaAgent:
    """One character in a conversation."""

    spec: PersonaSpec
    llm: LLM
    memory: str = ""  # bullet list from MemoryStore.recall(); empty means none
    window: int = DEFAULT_WINDOW

    @property
    def name(self) -> str:
        return self.spec.name

    def system_prompt(self) -> str:
        """The persona markdown and any memories, wrapped in in-character rules."""
        memory = _MEMORY_BLOCK.format(entries=self.memory) if self.memory.strip() else ""
        return _IN_CHARACTER_TEMPLATE.format(
            persona=self.spec.body,
            name=self.spec.name,
            memory=memory,
            pass_token=PASS_TOKEN,
            end_token=END_TOKEN,
        )

    def respond(self, transcript: List["Turn"], max_tokens: int = 1024) -> str:
        """
        Produce this character's next line (raw, control tokens included).

        The transcript is rewritten from this agent's point of view: its own past
        lines become `assistant` turns and everyone else's become `user` turns.
        That is what makes the model feel like a participant rather than an
        observer reading a script.
        """
        visible = list(transcript)
        if len(visible) > self.window:
            # Keep the scene-setting line; drop the oldest dialogue after it.
            visible = visible[:1] + visible[-(self.window - 1):]

        messages: List[dict] = []
        for turn in visible:
            if turn.speaker == self.name:
                messages.append({"role": "assistant", "content": turn.text})
            else:
                messages.append({"role": "user", "content": f"{turn.speaker}: {turn.text}"})

        # The API requires the first message to be from the user.
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(begin the conversation)"})
        # If everyone else stayed silent, the last message is our own line. Sent
        # as-is the API would treat it as a prefill and we would finish our own
        # sentence, so make the silence explicit instead.
        if messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "(silence — nobody else speaks)"})

        return self.llm.complete(self.system_prompt(), messages, max_tokens=max_tokens)


def agent_from_spec(spec: PersonaSpec, llm: LLM, memory: str = "") -> PersonaAgent:
    """Convenience constructor, so callers don't import the dataclass directly."""
    return PersonaAgent(spec=spec, llm=llm, memory=memory)
