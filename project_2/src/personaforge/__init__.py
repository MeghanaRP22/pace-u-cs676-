"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in five lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    convo = run_conversation([doctor, patient], topic="the test results")
    verdict = evaluate(convo, llm)          # an observer judges it

Or let the agent do all of it from plain English:

    Orchestrator(ClaudeLLM()).handle("make two doctors and a nurse and have them argue")

A persona is a markdown file. An agent is that file plus a model. A
conversation is agents taking turns. An observer judges the result, and each
persona remembers it next time.

Versioning: MAJOR.MINOR.PATCH. MINOR goes up when a public name is added,
MAJOR when one in `__all__` changes or goes away.
"""

from .agent import PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .errors import AmbiguousPersonaError, LLMError, PersonaForgeError, PersonaNotFoundError
from .llm import LLM, ClaudeLLM, DemoLLM, ScriptedLLM, ScriptedToolLLM, ToolCall, ToolTurn, default_llm
from .memory import MemoryStore
from .observer import Evaluation, evaluate
from .orchestrator import TOOLS, Orchestrator, Result
from .persona import (
    PersonaSpec,
    edit_persona,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    require_persona,
    save_persona,
)

__version__ = "0.2.0"

__all__ = [
    # personas
    "PersonaSpec",
    "generate_persona",
    "edit_persona",
    "save_persona",
    "load_persona",
    "list_personas",
    "find_persona",
    "require_persona",
    # agents and conversations
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    # observer and memory
    "Evaluation",
    "evaluate",
    "MemoryStore",
    # the agent you talk to
    "Orchestrator",
    "Result",
    "TOOLS",
    # models
    "LLM",
    "ClaudeLLM",
    "DemoLLM",
    "ScriptedLLM",
    "ScriptedToolLLM",
    "ToolCall",
    "ToolTurn",
    "default_llm",
    # errors
    "PersonaForgeError",
    "LLMError",
    "PersonaNotFoundError",
    "AmbiguousPersonaError",
]
