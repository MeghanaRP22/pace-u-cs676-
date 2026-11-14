"""
ui.py — the terminal interface.

Thin on purpose. Everything here is presentation: no persona logic, no model
calls. If you want to swap the terminal for a web app later, this is the only
module you should have to replace.
"""

from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

# Colours are assigned to speakers on first sight so each character keeps a
# consistent colour for the whole session.
_PALETTE = ["cyan", "magenta", "green", "yellow", "blue", "red"]
_assigned: Dict[str, str] = {}


def speaker_colour(name: str) -> str:
    """Stable colour for a speaker, assigned in order of first appearance."""
    if name not in _assigned:
        _assigned[name] = _PALETTE[len(_assigned) % len(_PALETTE)]
    return _assigned[name]


def banner(offline: bool, persona_dir: str) -> None:
    """Startup panel: what this is and how to drive it."""
    mode = "[yellow]OFFLINE[/yellow] (scripted replies, no API calls)" if offline else "[green]LIVE[/green] (Claude)"
    console.print(
        Panel(
            f"[bold]PersonaForge[/bold] — agent-to-agent persona simulation\n\n"
            f"Mode: {mode}\n"
            f"Personas: [dim]{persona_dir}/[/dim]\n\n"
            "[bold]Just say what you want, e.g.:[/bold]\n"
            "  make a patient with chronic back pain who distrusts doctors\n"
            "  and a blunt internist running forty minutes late\n"
            "  have them talk about the MRI results for 8 turns\n"
            "  make the doctor more patient\n\n"
            "[dim]/help for commands · /quit to exit[/dim]",
            border_style="blue",
            title="CS676 Project 2",
        )
    )


def help_text() -> None:
    """Command reference."""
    console.print(
        Panel(
            "[bold]Say it however you like.[/bold] In live mode Claude picks the tools:\n"
            "  create · list · show · edit personas, start a conversation\n\n"
            "  two doctors and a nurse, then have them argue about the diagnosis\n"
            "  show me the patient\n"
            "  make Maria less trusting\n"
            "  start a conversation between Maria and Dr Reyes about surgery for 10 turns\n\n"
            "[bold]Slash commands[/bold]\n"
            "  /help    this message\n"
            "  /cost    API calls and tokens used this session\n"
            "  /reset   forget the chat history (personas stay on disk)\n"
            "  /clear   clear the screen\n"
            "  /quit    exit",
            border_style="dim",
            title="help",
        )
    )


def user_prompt() -> str:
    """Read one line from the user."""
    return console.input("\n[bold blue]you[/bold blue] › ").strip()


def agent_says(text: str) -> None:
    """The orchestrator speaking to the user."""
    console.print("\n[bold blue]stage[/bold blue] › ", end="")
    console.print(Markdown(text))


def dialogue_line(speaker: str, text: str) -> None:
    """One line of persona-to-persona dialogue, printed as it arrives."""
    colour = speaker_colour(speaker)
    line = Text()
    line.append(f"{speaker}: ", style=f"bold {colour}")
    line.append(text)
    console.print(line)


def event(message: str) -> None:
    """Side effects worth surfacing — files written, mostly."""
    console.print(f"  [dim]· {message}[/dim]")


def error(message: str) -> None:
    console.print(f"[bold red]error[/bold red] {message}")


def conversation_header(topic: str) -> None:
    console.print(f"\n[dim]── conversation: {topic} ──[/dim]\n")


def evaluation(ev: Optional[object]) -> None:
    """The observer's verdict, shown under the conversation."""
    if ev is None:
        return
    score = f"{ev.goal_met}/5" if ev.goal_met is not None else "unscored"
    lines = [f"[bold]Observer[/bold] · goal {score}", ev.summary]
    if ev.goal_reason:
        lines.append(f"[dim]{ev.goal_reason}[/dim]")
    for flag in ev.all_flags:
        lines.append(f"[yellow]⚑[/yellow] {flag}")
    console.print(Panel("\n".join(lines), border_style="magenta"))


def usage(stats: Optional[Dict[str, int]]) -> None:
    """Running cost of the session, as counted by ClaudeLLM."""
    if not stats:
        console.print("[dim]No usage to report (offline mode makes no API calls).[/dim]")
        return
    console.print(
        f"[dim]{stats['calls']} API call(s) · {stats['input_tokens']:,} input tokens · "
        f"{stats['output_tokens']:,} output tokens[/dim]"
    )


def details(markdown: str) -> None:
    """Content shown verbatim: a persona list or a character sheet."""
    console.print(Panel(Markdown(markdown), border_style="dim"))
