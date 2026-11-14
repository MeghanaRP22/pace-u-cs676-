"""
persona.py — a persona is a markdown file on disk.

That is the whole idea of this package. There is no persona database and no
in-memory registry. A persona is a `.md` file in `temp/`, and anything that can
read that file can wake the persona up. You can open one in a text editor,
change a sentence, and the agent behaves differently on the next run.

File format — frontmatter for the machine, markdown for the model:

    ---
    name: Maria Delgado
    role: patient
    summary: 58-year-old with poorly controlled type 2 diabetes
    ---

    # Maria Delgado

    ## Background
    ...

Design rule for the parser: be strict about the three things the code indexes
on (name, role, summary) and generous about everything else. A model that drops
the closing `---`, wraps the document in a code fence, or quotes a value has
still told us who the character is, so we accept it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .errors import AmbiguousPersonaError, PersonaNotFoundError
from .llm import LLM

# Where persona files live. Relative to wherever you launched the app.
DEFAULT_PERSONA_DIR = Path("temp")

# Sections the author prompt asks for. A body with fewer than two of them is
# almost always the model talking *about* a persona instead of writing one; a
# body missing just one is a perfectly usable character, so we let it through.
EXPECTED_SECTIONS = ("Background", "Personality", "How they speak", "What they want", "What they avoid")
MIN_SECTIONS = 2


@dataclass
class PersonaSpec:
    """One persona: the metadata we index on, plus the markdown that is its mind."""

    name: str
    role: str
    summary: str
    body: str  # markdown; becomes the agent's system prompt
    # Where the spec was loaded from. Not part of equality: two specs with the
    # same content are the same character wherever they sit on disk.
    path: Optional[Path] = field(default=None, compare=False, repr=False)

    def slug(self) -> str:
        """Filesystem-safe stem, e.g. 'Maria Delgado' -> 'maria-delgado'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s or "unnamed"

    @property
    def file_id(self) -> str:
        """The stem of the file on disk, which is unique even when names collide."""
        return self.path.stem if self.path is not None else self.slug()

    def to_markdown(self) -> str:
        """Serialize to the frontmatter + body format shown above."""
        return (
            "---\n"
            f"name: {_quote_if_needed(self.name)}\n"
            f"role: {_quote_if_needed(self.role)}\n"
            f"summary: {_quote_if_needed(self.summary)}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "PersonaSpec":
        """
        Parse a persona file back into a spec.

        Missing frontmatter is tolerated: we fall back to sensible defaults and
        treat the whole file as the body, so a hand-written persona still loads.
        """
        meta = {"name": "Unnamed", "role": "unknown", "summary": ""}
        text = text.strip()

        # Models sometimes wrap the whole document in a code fence.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            text = text.strip()

        body = text

        if text.startswith("---"):
            rest = text[3:].lstrip("\n")
            closing = re.search(r"^---\s*$", rest, re.MULTILINE)
            if closing:
                # Well-formed: frontmatter is everything up to the closing fence.
                front, body = rest[: closing.start()], rest[closing.end():]
            else:
                # The closing '---' is missing, which the model does drop from
                # time to time. Read `key: value` lines from the top and treat
                # the first line that is not one as the start of the body.
                front_lines, body_lines, in_front = [], [], True
                for line in rest.splitlines():
                    if in_front and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:", line):
                        front_lines.append(line)
                    elif in_front and not line.strip():
                        continue  # blank line inside the header block
                    else:
                        in_front = False
                        body_lines.append(line)
                front, body = "\n".join(front_lines), "\n".join(body_lines)

            for line in front.splitlines():
                if ":" in line:
                    # partition() splits on the FIRST colon only, so a summary
                    # like "nurse: night shift" keeps its own colon.
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in meta:
                        meta[key] = _unquote(value.strip())

        return cls(name=meta["name"], role=meta["role"], summary=meta["summary"], body=body.strip())


def _quote_if_needed(value: str) -> str:
    """Quote a frontmatter value only when leaving it bare would be ambiguous."""
    if value and (value[0] in "\"'" or value != value.strip()):
        return '"' + value.replace('"', '\\"') + '"'
    return value


def _unquote(value: str) -> str:
    """Strip one matching pair of surrounding quotes, as YAML would."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].replace('\\"', '"')
    return value


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def normalize_role(role: str) -> str:
    """Reduce a role to one lowercase word: 'Emergency Physician' -> 'physician'."""
    words = re.findall(r"[a-z]+", role.lower())
    # The last word is usually the noun ("emergency physician", "ER nurse").
    return words[-1] if words else "person"


def validation_problems(spec: PersonaSpec) -> List[str]:
    """
    What is wrong with a persona the model wrote, as human-readable strings.

    An empty list means the spec is usable as-is. We only check what actually
    breaks the app: a missing name, or a body that is not a character sheet.
    """
    problems = []
    if spec.name in ("", "Unnamed"):
        problems.append("no name in the frontmatter")
    elif len(spec.name.split()) > 6 or re.search(r"\d", spec.name):
        problems.append(f"'{spec.name}' does not look like a name")
    present = [s for s in EXPECTED_SECTIONS if f"## {s}".lower() in spec.body.lower()]
    if len(present) < MIN_SECTIONS:
        problems.append("the body is not a character sheet (expected sections like ## Background)")
    return problems


# -----------------------------------------------------------------------------
# Disk
# -----------------------------------------------------------------------------


def save_persona(
    spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR, overwrite: bool = False
) -> Path:
    """
    Write the persona to `<directory>/<slug>.md`, creating the folder if needed.

    By default an existing file is never clobbered: a second "Maria Delgado"
    lands in `maria-delgado-2.md`. Pass `overwrite=True` to edit in place.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if overwrite and spec.path is not None:
        path = Path(spec.path)
    else:
        path = directory / f"{spec.slug()}.md"
        n = 2
        while path.exists() and not overwrite:
            path = directory / f"{spec.slug()}-{n}.md"
            n += 1

    path.write_text(spec.to_markdown(), encoding="utf-8")
    spec.path = path
    return path


def load_persona(path: Path) -> PersonaSpec:
    """Read one persona file from disk."""
    path = Path(path)
    spec = PersonaSpec.from_markdown(path.read_text(encoding="utf-8"))
    spec.path = path
    return spec


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> List[PersonaSpec]:
    """Every persona currently saved, sorted by filename. Empty if none yet."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [load_persona(p) for p in sorted(directory.glob("*.md"))]


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# Words people put in front of a name that carry no identity: "the doctor".
_FILLER = {"the", "a", "an", "dr", "mr", "mrs", "ms"}


def find_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> Optional[PersonaSpec]:
    """
    Look a persona up by name, file name, or role — case-insensitively.

    Matching happens in tiers and the first tier with any hit wins:

        1. exact file name                       "maria-delgado-2"
        2. exact full name or slug               "Maria Delgado"
        3. exact role                            "patient"
        4. every typed word starts a name word   "maria", "dr reyes", "del"
        5. role prefix                           "doc" -> doctor

    Whole-word prefixes mean "doc" can no longer match "Dorothy" by substring.
    If a tier has more than one hit we raise AmbiguousPersonaError instead of
    silently picking the first file — the caller should ask the user.

    :return: the spec, or None if nothing matched at all.
    """
    specs = list_personas(directory)
    needle = name.strip().lower()
    typed = [w for w in _words(needle) if w not in _FILLER]
    if not typed:
        return None
    joined = "-".join(typed)

    tiers = [
        lambda s: s.file_id in (needle, joined),
        lambda s: needle in (s.name.lower(), s.slug()) or joined == s.slug(),
        lambda s: joined == s.role.lower(),
        lambda s: all(any(w.startswith(t) for w in _words(s.name)) for t in typed),
        lambda s: len(typed) == 1 and len(typed[0]) >= 3 and s.role.lower().startswith(typed[0]),
    ]
    for matches in tiers:
        hits = [s for s in specs if matches(s)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise AmbiguousPersonaError(name, [f"{s.name} ({s.file_id})" for s in hits])
    return None


def require_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> PersonaSpec:
    """Like find_persona, but raises PersonaNotFoundError with the list of options."""
    spec = find_persona(name, directory)
    if spec is None:
        raise PersonaNotFoundError(name, [s.name for s in list_personas(directory)])
    return spec


# -----------------------------------------------------------------------------
# Generating and editing a persona with the model
# -----------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write character specifications for a multi-agent simulation.

Given a short description, write a persona as a markdown document. Output ONLY the
markdown — no preamble, no code fences.

Use exactly this structure:

---
name: <a specific, realistic full name>
role: <one lowercase word, e.g. patient, doctor, teacher, customer>
summary: <one line, under 15 words>
---

# <name>

## Background
Two or three sentences. Be concrete and specific — age, situation, history.

## Personality
Three or four bullet points about temperament and outlook.

## How they speak
Two or three bullets on voice: vocabulary, sentence length, verbal habits.

## What they want
One or two sentences on their goal in a conversation.

## What they avoid
One or two sentences on what they will not say or do.

Write a specific individual, not a type. Give them a quirk. Avoid stereotypes,
and do not make them a spokesperson for a demographic."""


def _ask_author(llm: LLM, request: str) -> PersonaSpec:
    """
    One authoring round trip with a single corrective retry.

    If the first reply fails validation, we tell the model exactly what was
    wrong and ask once more. One retry fixes the common failures (prose
    preamble, missing section) and caps the cost at two calls.
    """
    messages = [{"role": "user", "content": request}]
    markdown = llm.complete(system=_AUTHOR_SYSTEM, messages=messages, max_tokens=2048)
    spec = PersonaSpec.from_markdown(markdown)

    problems = validation_problems(spec)
    if problems:
        messages += [
            {"role": "assistant", "content": markdown},
            {
                "role": "user",
                "content": "That does not follow the required structure ("
                + "; ".join(problems)
                + "). Rewrite it, output only the markdown document.",
            },
        ]
        retry = PersonaSpec.from_markdown(llm.complete(_AUTHOR_SYSTEM, messages, max_tokens=2048))
        if len(validation_problems(retry)) < len(problems):
            spec = retry

    spec.role = normalize_role(spec.role) if spec.role != "unknown" else spec.role
    return spec


def generate_persona(description: str, llm: LLM) -> PersonaSpec:
    """
    Ask the model to author a persona from a one-line description.

    :param description: e.g. "a patient with type 2 diabetes who distrusts doctors"
    :param llm:         any LLM implementation (real or scripted)
    :return:            the parsed PersonaSpec — not yet saved to disk
    """
    spec = _ask_author(llm, f"Write a persona for: {description}")

    # If the model ignored the format even after the retry we still want
    # something usable, so fall back to the raw description.
    if spec.name == "Unnamed":
        spec.name = description[:40].strip().title() or "Unnamed"
        spec.summary = description.strip()
    return spec


def edit_persona(spec: PersonaSpec, instruction: str, llm: LLM) -> PersonaSpec:
    """
    Rewrite an existing persona according to an instruction, keeping its name.

    "make the doctor more skeptical" becomes one model call that returns the
    whole revised document. The name is pinned so memories and transcripts
    that refer to this character stay attached to it.
    """
    request = (
        "Here is an existing persona:\n\n"
        f"{spec.to_markdown()}\n\n"
        f"Revise it: {instruction}\n"
        f"Keep the name exactly '{spec.name}'. Return the full revised document."
    )
    revised = _ask_author(llm, request)
    if validation_problems(revised):
        # A broken revision must not overwrite a working persona.
        return spec
    revised.name = spec.name
    revised.path = spec.path
    return revised


# =============================================================================
# STATUS OF THE ORIGINAL "YOUR TASK" LIST
# =============================================================================
#
#  1. find_persona disambiguation      DONE — tiered whole-word matching, raises
#                                      AmbiguousPersonaError instead of guessing.
#  2. save_persona overwrites          DONE — collisions get a -2, -3 suffix.
#  3. frontmatter quoting / colons     DONE — first-colon split, quote stripping.
#  4. validation of model output       DONE — validation_problems() + one retry.
#  5. editing from inside the app      DONE — edit_persona(), exposed as a tool.
#  6. personas changing over time      Handled in memory.py, deliberately NOT by
#                                      rewriting this file. See REPORT.md.
