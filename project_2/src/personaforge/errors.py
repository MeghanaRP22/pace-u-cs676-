"""
errors.py — every exception this package raises on purpose.

One base class, so a caller can write `except PersonaForgeError` and catch
everything we meant to raise while letting genuine bugs (a TypeError, say)
propagate. Each message says what to do next, not just what went wrong.
"""

from __future__ import annotations

from typing import List


class PersonaForgeError(Exception):
    """Base class for all errors raised deliberately by personaforge."""


class LLMError(PersonaForgeError):
    """A model call failed: bad key, network down, rate limit, or server error."""


class PersonaNotFoundError(PersonaForgeError):
    """No saved persona matched the name the user gave."""

    def __init__(self, name: str, available: List[str]) -> None:
        self.name = name
        self.available = available
        hint = f" Available: {', '.join(available)}." if available else " No personas exist yet."
        super().__init__(f"I couldn't find a persona matching '{name}'.{hint}")


class AmbiguousPersonaError(PersonaForgeError):
    """More than one saved persona matched equally well, so we refuse to guess."""

    def __init__(self, name: str, candidates: List[str]) -> None:
        self.name = name
        self.candidates = candidates
        super().__init__(
            f"'{name}' matches more than one persona: {', '.join(candidates)}. "
            "Use the full name or the file name to pick one."
        )
