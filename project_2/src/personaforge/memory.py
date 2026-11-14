"""
memory.py — what a persona remembers between conversations.

Where memory lives is a design decision, so here it is stated plainly:

    temp/maria-delgado.md            who Maria IS       (authored, edited by you)
    temp/memory/maria-delgado.md     what Maria has DONE (appended by the app)

Memory goes in a separate file, one per persona, and never back into the
persona file itself. Three reasons:

  1. Identity should not drift. If every conversation appended to the persona
     file, the character sheet you wrote would slowly be buried under the
     model's own summaries, and after twenty runs the model would be reading
     mostly its own output. Keeping them apart means deleting a memory file
     resets a character without touching who they are.
  2. Privacy between agents. Each agent is shown only its own memory file. The
     doctor does not learn what the patient remembers, which is how real
     participants work (an open question in the original agent.py).
  3. Bounded cost. Only the last few entries are injected into the prompt, so
     a persona that has had fifty conversations costs the same per turn as
     one that has had five.

It is still just markdown, so you can open it, read it, and edit it — the same
contract as the persona file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from .persona import PersonaSpec

DEFAULT_RECALL = 5


class MemoryStore:
    """Reads and appends per-persona memory files under `<persona_dir>/memory/`."""

    def __init__(self, persona_dir: Path, max_entries: int = DEFAULT_RECALL) -> None:
        self.directory = Path(persona_dir) / "memory"
        self.max_entries = max_entries

    def path_for(self, spec: PersonaSpec) -> Path:
        """One file per persona, keyed on the persona's own file name."""
        return self.directory / f"{spec.file_id}.md"

    def entries(self, spec: PersonaSpec) -> List[str]:
        """Every remembered entry, oldest first. Empty if this persona has none."""
        path = self.path_for(spec)
        if not path.exists():
            return []
        return [line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]

    def recall(self, spec: PersonaSpec) -> str:
        """The most recent entries as a bullet list, ready for a system prompt."""
        recent = self.entries(spec)[-self.max_entries:]
        return "\n".join(f"- {e}" for e in recent)

    def remember(self, spec: PersonaSpec, entry: str, when: Optional[date] = None) -> Path:
        """Append one entry. Newlines are flattened so one entry is one bullet."""
        path = self.path_for(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"# Memories of {spec.name}\n\n", encoding="utf-8")
        stamp = (when or date.today()).isoformat()
        line = " ".join(entry.split())
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {stamp}: {line}\n")
        return path

    def forget(self, spec: PersonaSpec) -> bool:
        """Delete this persona's memory. Returns True if there was anything to delete."""
        path = self.path_for(spec)
        if path.exists():
            path.unlink()
            return True
        return False


def memory_entry(speaker: str, participants: List[str], topic: str, summary: str) -> str:
    """
    Phrase one memory from a participant's own point of view.

    "Talked with Dr. Reyes about the MRI results. <summary>" reads naturally
    when it is injected into that participant's prompt next time.
    """
    others = [p for p in participants if p != speaker]
    with_whom = ", ".join(others) if others else "nobody"
    return f"Talked with {with_whom} about {topic}. {summary}".strip()
