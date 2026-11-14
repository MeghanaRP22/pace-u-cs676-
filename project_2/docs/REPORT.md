# PersonaForge 0.2.0 — Technical Report

CS676 Algorithms for Data Science · Project 2

> **Before submitting:** fill in the three places marked **✍️ TODO** (name, the live
> transcript, and your live cost numbers). Everything else describes code that is in
> this repository and is covered by the test suite.

**Author:** ✍️ TODO your name · claude-haiku-4-5 ✍️ 

---

## 1. Scope, and why

The assignment offers five extensions and says depth beats breadth. I went
**deep on #1 (tool use)** because it is the one that changes what the program *is*.
A command parser becomes an agent. I then built #3, #4 and #2 **as a single
pipeline**, because they share infrastructure and are much cheaper together than
apart:

| # | Extension | Depth | Where |
|---|---|---|---|
| 1 | Regex router → real tool use | **Full.** 5 tools, multi-call turns, clarifying questions, errors returned to the model, loop cap, safe history trimming | `orchestrator.py`, `llm.py` |
| 3 | Interesting turn-taking | Addressed-speaker rule, `[PASS]`, `[END]`, silence detection, survives API failure | `conversation.py`, `agent.py` |
| 4 | Observer | Two layers: free rule checks + one model-judge call | `observer.py` |
| 2 | Memory | Per-persona private memory file, bounded recall | `memory.py` |
| 5 | Package quality | Error hierarchy, public API, semantic version, usage/cost tracking | `errors.py`, `__init__.py` |

I also fixed every item on `persona.py`'s YOUR TASK list, because the tool-use agent
made those bugs much easier to hit. With tools, people refer to personas by
whatever name they like ("the doctor", "Dr Reyes", "maria"), and the old
substring matcher guessed wrong without saying so.

What connects #2, #3 and #4: the observer's one-paragraph summary **is** the
memory each participant keeps. One extra model call per conversation pays for
both features.

---

## 2. Architecture

```
            you type anything
                   │
                   ▼
   ┌──────────────────────────────────┐     tools=[create_persona, list_personas,
   │ Orchestrator.handle()            │◄──── show_persona, edit_persona,
   │  supports_tools(llm)?            │      start_conversation]
   │   yes → tool-use loop (Claude)   │
   │   no  → router.plan() (patterns) │
   └──────────────┬───────────────────┘
                  │ (tool_name, args)  ← same shape from both paths
                  ▼
          _run_tool()  ── the ONE executor; PersonaForgeError → is_error tool_result
                  │
   ┌──────────────┼──────────────────────────────────────────────┐
   ▼              ▼                                              ▼
persona.py    run_conversation()                             observer.evaluate()
 generate →    next_speaker(): addressed > quiet-longest      rule_flags()  (free)
 validate →    agent.respond()  ← persona.md + memory          model judge   (1 call)
 retry once    parse_control(): [PASS] / [END]                     │
 save (no      on failure: keep transcript, record error           ▼
 overwrite)              │                                  memory.remember()
                         ▼                                   temp/memory/<persona>.md
             temp/conversations/<time>-<topic>.md
```

**Files on disk** (all plain markdown, all editable by hand):

```
temp/
├── maria-delgado.md               who Maria IS        (you or the model write it)
├── memory/maria-delgado.md        what Maria has DONE (the app appends to it)
└── conversations/20261113-…md     transcript + observer verdict
```

---

## 3. Design decisions

### 3.1 Tool use (the headline)

**The loop.** `_handle_with_tools` sends the user's message along with the five tool
schemas. While the reply contains `tool_use` blocks, it runs each tool, appends
the matching `tool_result` blocks, and asks again. A plain-text reply ends the
loop. That one structure gives three behaviours the regex router could not:

| Request | Old regex router | Now |
|---|---|---|
| `create a skeptical data scientist named Ada` | "I'm a stub" (no match) | `create_persona` |
| `create a persona` | canned string | the model asks *"What kind of person?"* (no tool call) |
| `make two doctors and a nurse and have them argue` | created one persona called "it two doctors…" | 3 × `create_persona` in one turn, then `start_conversation` |
| `list the personas then have them talk` | listed only (pattern order) | both, in order |

**Errors go to the model, not to a stack trace.** An unknown or ambiguous persona
raises `PersonaNotFoundError` or `AmbiguousPersonaError`. The executor turns that
into a `tool_result` with `is_error: true`, which tells Claude exactly what exists
("Available: Ann Lee, Bo Kim"). The model then asks the user a sensible follow-up
instead of the turn crashing. This is what lets the agent *clarify*.

**One executor.** Whether a tool was chosen by Claude or by the fallback pattern
planner, it runs through `_run_tool()`. So a tool behaves identically on both
paths, and the 25 orchestrator tests cover both.

**Why keep the regexes at all?** They were demoted from router to fallback
(`router.py`) for two jobs: models that cannot call tools (`ScriptedLLM`, and so
the original tests still pass unchanged), and **standing in for Claude inside
`DemoLLM`**. The second job means `--offline` now exercises the *real* tool loop,
with tool_use IDs, tool_result echo and multi-step plans, for free. I made the
planner noticeably better than the original (it handles "then", counts, and "N
turns") so the offline demo is honest about the flow, even though its
*understanding* is just patterns.

**Guard rails.**
- `MAX_TOOL_ROUNDS = 8` stops a model that keeps calling tools.
- `max_turns` is clamped to 2–20, and a non-integer value falls back to the default.
- History is trimmed to 30 messages, but **only at a plain user message**. Cutting
  between a `tool_use` and its `tool_result` makes the API reject every later request.
- "Have them talk" with no names uses the personas created in this same request,
  and refuses to guess when more than two exist. That fixes the skeleton bug where
  it silently picked the first two files alphabetically.
- Lists and character sheets go to the user **verbatim** (`Result.details`), never
  paraphrased by the model.

### 3.2 Turn-taking: a rule, not a model

*Who decides who speaks next?* I chose a **deterministic rule** over a moderator
model:

1. Nobody speaks twice in a row.
2. If the last line names exactly one other participant, by first name, surname or
   role ("Nurse, the chart please"), that participant answers.
3. Otherwise, whoever has been quiet longest speaks.

With two people this reduces to alternation, so nothing changed for the common
case. A moderator model would cost one extra call *per line*, which doubles the
price of every conversation, and it would make tests non-deterministic. The rule
captures the main thing real conversations do: answer whoever was addressed.

**Agents can now stay silent or leave.** The in-character prompt tells each agent
it may reply `[PASS]` or end its line with `[END]`. `parse_control()` is generous
about format (`[pass].`, `[END]` on its own line) and treats an empty reply as a
pass. The loop ends when:
- an agent says `[END]`, but only after **everyone** has spoken, so an eager
  opener cannot close the room alone;
- every participant passes in a row (the room went quiet);
- the `max_turns` ceiling is reached.

**Failure no longer loses the conversation.** If a model call raises
mid-conversation, the lines already spoken are kept. `Conversation.error` and
`ended_by` record what happened, and the partial transcript is still saved and
judged.

**Cost control.** Each agent sees the scene line plus a window of the last 20 lines
(`agent.window`). Without it, every turn re-sends the whole history, and total
tokens grow quadratically with conversation length.

### 3.3 The observer: two layers

| Layer | Catches | Cost |
|---|---|---|
| `rule_flags()` | "as an AI"/"simulation", `*stage directions*`, writing another character's line (`Bo Kim: …` inside Ann's turn), verbatim repetition, cut-short conversations | free, deterministic, fully tested |
| model judge | goal achieved (1–5 with reason), per-character in-character verdict, summary, softer problems (lost the thread, never answered) | one call |

The judge is asked for JSON. Following the README's lesson ("be strict about what
matters, generous about everything else"), `parse_evaluation()` extracts the first
`{…}` even from inside a code fence or after a preamble. It accepts `"5"` as 5 and
rejects scores outside 1–5. It drops `in_character` entries for names that are not
participants. If the reply is prose, that prose becomes the summary. A failing
judge never costs the user the conversation they just watched.

### 3.4 Memory: a separate file, never the persona file

The README asked me to defend this choice. Memory lives in
`temp/memory/<persona>.md`, **not** appended to the persona's own `.md`:

1. **Identity must not drift.** If every run appended to the persona file, the
   character sheet you wrote would slowly be buried under the model's summaries.
   After twenty runs the model would mostly be reading its own output, a feedback
   loop. Kept apart, deleting one memory file resets a character's history without
   touching who they are.
2. **Privacy between agents.** Each agent is shown only its own memory file, so the
   doctor does not get the patient's memories (agent.py YOUR TASK #5).
3. **Bounded cost.** Only the last 5 entries enter the prompt, so a persona with 50
   past conversations costs the same per turn as one with 5.

It is still markdown you can open and edit, the same contract as the persona file.

### 3.5 Persona hardening

| Skeleton problem | Fix |
|---|---|
| `"doc"` matched `"Dorothy"`; equal roles were indistinguishable | 5-tier whole-word matching (file name → full name → role → name-word prefixes → role prefix). A tie raises `AmbiguousPersonaError` listing the candidates |
| A second "Maria Delgado" overwrote the first | `maria-delgado-2.md`; `overwrite=True` only for edits |
| Quoting not handled | Quotes stripped on read, added on write when needed; colons inside values kept (split on the first colon) |
| No validation of model output | `validation_problems()`, then **one** corrective retry that tells the model what was wrong. Strict on name and on "is this a character sheet at all", relaxed on which sections are present |
| Roles like "Emergency Physician" | Normalised to one lowercase word (`physician`) |
| No in-app editing | `edit_persona` tool. The name is pinned so memories stay attached, and a malformed revision never overwrites a working file |
| Transcripts saved as `temp/conversation.md` were later **listed as a persona called "Unnamed"** | Transcripts moved to `temp/conversations/` with unique timestamped names |

Two characters with the same name would each treat the other's lines as their own,
because an agent recognises itself by name. The orchestrator therefore
disambiguates display names at conversation time ("Ann Lee (ann-lee-2)").

### 3.6 Package quality

- **Errors that say what to do next:** `PersonaForgeError` is the base class, with
  `LLMError` ("Your API key was rejected. Check ANTHROPIC_API_KEY in .env."),
  `PersonaNotFoundError` and `AmbiguousPersonaError`. SDK exceptions (401, 404, 429,
  5xx, connection) are translated in one place, `ClaudeLLM._create`.
- **Public API:** `__all__` is grouped by concept. `ScriptedToolLLM` and `ToolTurn`
  are public so *users* of the package can test their own orchestrations offline.
- **Version 0.2.0:** MINOR was bumped because public names were added. The rule is
  written in the package docstring.
- **Cost is visible:** `ClaudeLLM.usage` counts calls and tokens. Type `/cost` in the
  terminal.
- **The UI stayed thin:** `cli.py` and `deploy/streamlit_app.py` contain no
  persona logic. The web front end was a single small file, which is the point
  of a package boundary.

---

## 4. A transcript, with commentary

### 4.1 Offline (reproducible, free)

`uv run main.py --offline`, then:
`make it two doctors and a nurse, and have them argue about the diagnosis for 6 turns`

The saved file is `docs/transcripts/offline-demo.md`. What it demonstrates:

- **One sentence → four tool calls.** Three `create_persona` calls and a
  `start_conversation`, through the real tool loop (DemoLLM standing in for Claude).
- **Name collisions handled.** The two doctors got the same stub name, so their
  files became `doctor.md` and `doctor-2.md`, and they appear in the conversation as
  "Doctor (doctor)" and "Doctor (doctor-2)" rather than confusing each other.
- **Three-way turn order.** The nurse got her turn. Under the skeleton's rule she
  would have, too, but now it is because she had been quiet longest, not because
  of list order.
- **Early ending.** The last scripted line ends with `[END]`, so `ended_by` reads
  "a participant closed it".
- **The observer ran** and its verdict is appended to the transcript.

What it does **not** show: any intelligence. The dialogue is canned, so the "nurse"
speaks patient lines. That is why §4.2 matters.

### 4.2 Live — ✍️ TODO

Run this (about 25–35 API calls with Haiku):

```bash
uv run main.py --model claude-haiku-4-5
```
```
a 58-year-old retired teacher with chronic back pain who distrusts doctors, and a blunt internist running forty minutes late
have them talk about the MRI results for 10 turns
/cost
have them talk about whether she should try physical therapy
/cost
```

------------

# Conversation: the MRI results

Participants: Margaret Reeves, Dr. Marcus Okonkwo

**Margaret Reeves:** You've just received a call from your doctor's office scheduling a follow-up appointment to discuss your lumbar spine MRI results. You're sitting in your reading chair with the phone pressed to your ear, feeling that familiar knot of dread.

---

Well, they won't tell me anything over the phone, naturally. Says the doctor needs to "walk me through the findings." Which I suspect means she wants to sell me on some expensive intervention I don't need. I've already looked up what they're probably going to find—herniated disc, maybe some stenosis. Everyone my age has these things. The question is whether you actually do anything about it, or whether you just live with it like most sensible people do. But I could be wrong about their motives, I suppose. They're probably just following protocol.

**Dr. Marcus Okonkwo:** Look, I appreciate the skepticism—honestly, most of it's warranted. But let me be straight with you: if my office called you in for a follow-up instead of just saying "everything's fine, see you next year," there's something on that MRI worth discussing. That's not protocol padding, that's just... there's something.

Now, whether that something requires surgery or injections or just stretching exercises and ice packs—that's what we need to talk through. You're right that plenty of people have disc bulges they never notice. But the ones we call patients about usually aren't in that category.

When's your appointment?

**Margaret Reeves:** I suppose that's fair enough. You're not wrong that if everything were perfectly normal, there wouldn't be much point in the conversation, would there?

My appointment is Thursday at two o'clock. Though I'll admit I've been dreading it all week—I've already spent more time than I care to admit reading about what they might find, which I'm sure isn't helping my state of mind. The internet is a marvelous invention for learning and a dreadful one for catastrophizing, but I could be wrong about that.

I just... I want to know what we're actually dealing with, not what some pharmaceutical company has decided is profitable to treat. And I want someone to tell me honestly whether this is something that will genuinely improve with intervention, or whether I'm just paying for the illusion of improvement. That's all I'm asking for.

**Dr. Marcus Okonkwo:** Thursday at two. Good. That's soon enough you don't have to sit with this dread much longer.

Look, I'm going to tell you something that might sound strange coming from a doctor: I'd rather tell you "this doesn't need surgery, here's what actually helps" than sell you on an intervention. Surgery means liability, means complications, means you come back angry. A patient who does their physical therapy and feels better? That's a win I actually sleep on.

The internet catastrophizing—yeah, that's real. You've probably read about the worst-case scenarios and convinced yourself that's what's happening. Doesn't help the dread, does it?

Here's what I'd suggest: go to that appointment Thursday, listen to what she actually says about the imaging, and ask her straight out—"If I do nothing, what happens?" and "If I do something, what's the realistic improvement?" Make her answer those specifically. If she hedges, that tells you something too.

And Margaret? You're not wrong to want honesty. You deserve it. Most of what you're dealing with right now is the not-knowing, not the thing itself.

**Margaret Reeves:** You know, you might be right about that. The not-knowing is rather like waiting for a student's standardized test scores—the anticipation is somehow worse than whatever the actual number turns out to be.

I appreciate what you've said about the liability question. That's actually useful to know. I suppose I've been operating under the assumption that every doctor has an incentive to intervene, but you're right—that cuts both ways, doesn't it? A failed surgery is far more costly than honest conservative management.

Those two questions you mentioned are good ones. "If I do nothing, what happens?" and "If I do something, what's the realistic improvement?" I'll write them down so I don't forget them in the moment, because I do have a tendency to become muddled when I'm anxious.

I'll go Thursday and actually listen instead of sitting there mentally cross-examining her. Though I make no promises about not reading more dreadful internet articles between now and then—old habits die hard. But I could be wrong about whether that's entirely a waste of time. At least I'll know the terminology she uses.

Thank you for that. For the honesty, I mean. And for not treating my skepticism like it's just paranoia

---

_Ended: a participant closed it after 5 lines._

## Observer

**Summary:** Margaret and Dr. Okonkwo had a productive conversation about her upcoming MRI follow-up appointment. Dr. Okonkwo reassured her that the call-in indicates a real finding worth discussing, validated her skepticism about over-intervention, and gave her concrete questions to ask her doctor on Thursday to determine necessity and realistic outcomes.

**Goal:** make real progress on: the MRI results — **4/5**. They made genuine progress identifying the core issue (anxiety about the unknown) and arming Margaret with a practical strategy for the actual appointment, though the MRI results themselves remain undiscussed since this was a pre-appointment phone call.

**In character:** Margaret Reeves ✓, Dr. Marcus Okonkwo ✓

**Flags:**
- none

Copy the second transcript from `temp/conversations/` to
`docs/transcripts/live-demo.md`, then write 1–2 paragraphs answering:

1. Did the first message become **two** `create_persona` calls (one sentence, two people)?
2. Does the **second** conversation reference the first? That is memory working. Quote the line.
3. Did anyone `[PASS]`, or did someone `[END]` before the limit? Was it the right moment?
4. Do you agree with the observer's score? Where would you have scored differently, and why?
5. One line where a character was most *in* character, and one where they slipped.

---

## 5. Failure analysis

These are failure modes I either observed while building it or can identify from
the design. Each one says what the code does about it and what it does not.

### 5.1 Breaking character

- **Lexical breaks are caught; tonal breaks are not, reliably.** `rule_flags()` sees
  "as an AI" and `*sighs*` with certainty. A doctor who suddenly sounds like a
  customer-service bot ("Great question! Here are three options:") contains no
  forbidden word. Only the model judge can notice it, and the judge is the same
  family of model that made the mistake. It may not flag a style it finds normal.
- **Speaking for others.** The rule check catches `Name:` prefixes, but not
  narrated speech ("…and then Maria would probably say she's fine").
- **Control tokens can leak.** If a model writes "[END" or "(pass)", it does not
  match the token. The line is shown as dialogue and the conversation runs on to
  the limit.

### 5.2 Losing the thread

- **The 20-line window.** In a long conversation an agent can forget something
  said 25 lines earlier, and only the scene line is pinned. A rolling summary
  would fix this at the cost of an extra call.
- **Addressed-speaker false positives.** Names that are common words ("Will",
  "Grace", "Hope") or a role used generically ("my *doctor* said…" said to a
  nurse) will route the next turn to the wrong person. The rule only acts when
  *exactly one* participant is matched, which limits the damage but does not
  prevent it.
- **Premature `[END]`.** A model may end as soon as its own question is answered,
  even if the other side's is not. The "everyone has spoken" guard is weak, since
  one line each is enough.

### 5.3 Memory

- **Memories are the observer's words, not the character's.** Both participants
  store the same third-person summary. Privacy holds between *files*, but the
  summary can include the observer's inference about the *other* person's
  feelings. A per-participant summary would be more realistic and would cost one
  call per participant.
- **Stale memories are never revised.** If you edit the doctor to be kinder, their
  memory still says they were curt last time. That is arguably realistic, and
  `MemoryStore.forget()` is the escape hatch.

### 5.4 Tool use

- **Cost of agency.** Every user message is at least two orchestrator calls (choose
  tools, then summarise results), each carrying the system prompt and five tool
  schemas. The regex router cost zero calls for commands.
- **The model can do the wrong thing confidently.** For example, it can create a
  persona when you meant to edit one. Tool results are reported back, so the user
  sees what happened, but nothing is undoable except by editing files.
- **Offline mode understands only patterns.** `--offline` proves the plumbing works,
  not the understanding. "Ada, a skeptical data scientist" without the word
  "create" gets a stub reply offline and works live.

### 5.5 Cost — ✍️ TODO: replace estimates with your `/cost` numbers

Estimated calls for one "create two people, then have them talk for N lines" session:

| Step | Calls | Notes |
|---|---|---|
| Orchestrator rounds | 2–3 per user message | choose tools → (maybe more tools) → summarise |
| Persona authoring | 1 each, 2 if the retry triggers | retry only on malformed output |
| Dialogue | N + passes | passes are calls that produce no line |
| Observer | 1 | skip with `--no-observer` |

For N = 10 that is roughly **17–20 calls**. Input tokens dominate. Each dialogue
call re-sends the persona (~600 tokens) and up to 20 lines, so cost grows with
window × turns. The window caps the growth at linear once a conversation passes
20 lines.

| Measured with `/cost` | Calls | Input tokens | Output tokens |
|---|---|---|---|
| ✍️ After conversation 1 | | | |
| ✍️ After conversation 2 | | | |


you › /cost
15 API call(s) · 19,127 input tokens · 2,475 output tokens

you › /quit


### 5.6 Deployment

- On Streamlit Cloud, visitors spend **the deployer's key**. Each session is capped
  at 15 live requests and defaults to Haiku. That is a speed bump, not real abuse
  protection: anyone can open a new session.
- Per-session temp folders are not deleted, and they vanish when the host
  restarts. That is acceptable for a demo, but it means web personas are not
  persistent.

---

## 6. Testing

**157 tests, all offline, ~2 s. Line coverage 98%** (`uv run pytest --cov=personaforge`).

| Module | Cover | Test file(s) |
|---|---|---|
| agent.py | 100% | test_agent, test_turn_taking |
| conversation.py | 99% | test_conversation, test_turn_taking |
| errors.py | 100% | (via all) |
| llm.py | 98% | test_llm, test_claude_and_cli |
| memory.py | 100% | test_observer_memory |
| observer.py | 99% | test_observer_memory |
| orchestrator.py | 99% | test_orchestrator_tools, test_router, test_conversation |
| persona.py | 100% | test_persona, test_persona_hardening |
| router.py | 97% | test_router |
| cli.py / ui.py | 94% / 96% | test_claude_and_cli |
| deploy/streamlit_app.py | smoke | test_streamlit_app (Streamlit `AppTest`, headless) |

How the fakes work:

- `ScriptedToolLLM` plays Claude in the tool loop. It has separate queues for tool
  turns and for plain completions, so a test can script "the model calls
  `create_persona` three times" without counting dialogue calls.
- `PerAgentLLM` (in the tests) gives each persona its own script, which is how the
  addressed-speaker and silence rules are tested exactly.
- `ClaudeLLM` is tested against a **fake SDK client** that returns SDK-shaped
  objects and raises real `anthropic` exception types. That covers request
  building, `tool_use` parsing, usage counting and every error translation,
  without a key.

**Not tested, and not testable offline:** prompt quality, whether Claude actually
chooses the right tools, and whether characters stay in character. Those are the
job of §4.2 and the observer.

Two bugs were **found by the new tests**, not by me reading the code:
1. A two-letter name ("Bo") was never recognised as being addressed.
2. The file name `ann-lee` was ambiguous when two personas shared a name, because
   both had the same slug.

---

## 7. Novelty (for the presentation)

1. **The observer pays twice.** One judge call yields a score for the user *and* a
   memory for every participant.
2. **The tool loop is fully testable and demoable without a key.** The fallback
   planner emits the same `(tool, args)` shape as Claude, and `DemoLLM` wraps it in
   real `tool_use` blocks. A live demo can fall back to offline mid-presentation
   and still show the agent loop.
3. **Errors are a conversation, not a crash.** Persona lookup failures become
   `is_error` tool results, so ambiguity turns into a clarifying question.
4. **A cheap turn-taking rule beats a costly moderator** for the common case, and
   the report says where it fails (§5.2).

---

## 8. If I had more time

- A model-driven moderator, used only for 3+ participants, to compare against the rule.
- A per-participant memory summary instead of a shared one.
- A rolling summary that replaces lines falling out of the window.
- Undo for `edit_persona` (keep `.bak` files).
- Streaming dialogue in the web UI token by token.
