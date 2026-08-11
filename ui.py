"""
ui.py -- Hey Neo terminal UI

Owns ALL visual rendering:
  - Matrix-decode intro animation (left-to-right, character-depth staging)
  - Thinking / tool-call trace panel with live reasoning token stream
  - Two-phase streaming: Phase A (trace) commits permanently, Phase B (response) follows
  - Input prompt with separator
  - Exit panel

Entry points:
  python main.py          -> real agent (build_agent + rewrite_query), live streaming
  python ui.py            -> same
  python ui.py --mock     -> demo only (no Ollama, animated mock)
"""

import sys
import time
import random
import copy
import json
from contextlib import contextmanager
from typing import TypedDict

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme
from rich import box

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
NEO_THEME = Theme({
    "neo.prompt":    "bold color(51)",
    "neo.tag":       "bold color(159)",
    "neo.hint":      "color(146) dim",
    "neo.exit":      "bold color(196)",
    "neo.think.hdr": "bold color(99)",
    "neo.think.txt": "color(153)",
    "neo.tool":      "bold color(214)",
    "neo.tool.arg":  "color(222) dim",
    "neo.ok":        "color(82)",
    "neo.err":       "color(203)",
    "neo.sep":       "color(237)",
    "neo.shadow":    "color(57) dim",
})

console = Console(theme=NEO_THEME, highlight=False)

# ─────────────────────────────────────────────────────────────────────────────
# Logo data
# ─────────────────────────────────────────────────────────────────────────────
_LOGO_LINES = [
    "  ███╗   ██╗███████╗ ██████╗ ",
    "  ████╗  ██║██╔════╝██╔═══██╗",
    "  ██╔██╗ ██║█████╗  ██║   ██║",
    "  ██║╚██╗██║██╔══╝  ██║   ██║",
    "  ██║ ╚████║███████╗╚██████╔╝",
    "  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ",
]
_TAGLINE = "  Your local Linux intelligence. Powered by Ollama."
_GLITCH  = list("█▓▒░▄▀■□▪▫◆◇○●▲▼◀▶╔╗╚╝║═╠╣╦╩╬┌┐└┘│─≡≈∑∞")

EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}

_DEPTH_STAGES = ["░", "▒", "▓"]
_SETTLED_CLR  = "bold color(51)"
_STAGE_CLR    = "color(51)"
_GLITCH_CLR   = "color(238)"


# ─────────────────────────────────────────────────────────────────────────────
# Intro animation
# ─────────────────────────────────────────────────────────────────────────────

def animate_intro() -> None:
    """3-phase animated intro: matrix-decode logo -> typewriter tagline -> hint panel."""
    FRAMES       = 36
    FRAME_DELAY  = 0.046
    STAGE_FRAMES = 3
    STAGE_WIDTH  = STAGE_FRAMES / FRAMES
    max_col = max(len(line) - 1 for line in _LOGO_LINES)

    def _frame(progress: float) -> Text:
        out = Text()
        for line in _LOGO_LINES:
            for col, ch in enumerate(line):
                if ch == " ":
                    out.append(" ")
                    continue
                reveal_at = col / max_col if max_col > 0 else 0.0
                if progress < reveal_at:
                    out.append(random.choice(_GLITCH), style=_GLITCH_CLR)
                else:
                    age       = progress - reveal_at
                    stage_idx = int(age / STAGE_WIDTH)
                    if stage_idx < len(_DEPTH_STAGES):
                        out.append(_DEPTH_STAGES[stage_idx], style=_STAGE_CLR)
                    else:
                        out.append(ch, style=_SETTLED_CLR)
            out.append("\n")
        return out

    with Live(_frame(0.0), console=console, refresh_per_second=30, transient=False) as live:
        for i in range(FRAMES + 1):
            live.update(_frame(i / FRAMES))
            time.sleep(FRAME_DELAY)

    sys.stdout.write("\n")
    for ch in _TAGLINE:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(0.028)
    sys.stdout.write("\n\n")
    sys.stdout.flush()

    console.print(
        Panel(
            "[neo.hint]Ask anything about your system -- packages, hardware, services, configs.\n"
            "Type [bold color(51)]/exit[/bold color(51)] to quit.[/neo.hint]",
            border_style="color(57)",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────────

def get_user_input() -> str:
    console.print(Rule(style="neo.sep"))
    return console.input("[neo.prompt]  You >[/neo.prompt]  ").strip()


def is_exit_command(text: str) -> bool:
    return text.lower() in EXIT_COMMANDS


# ─────────────────────────────────────────────────────────────────────────────
# Thinking trace
# ─────────────────────────────────────────────────────────────────────────────

_STEP_STYLE = {
    "think": ("💭", "neo.think.txt"),
    "tool":  ("⚡", "neo.tool"),
    "done":  ("❆", "neo.ok"),
    "error": ("✗", "neo.err"),
}


class _MockStep(TypedDict, total=False):
    type:   str
    label:  str
    detail: str
    _delay: float


_MAX_REASON = 500  # sliding window: show last N chars of reasoning text


def _build_thinking(steps: list, reasoning: str = "") -> Text:
    out = Text()
    out.append("  ◆ ", style="neo.think.hdr")
    out.append("THINKING\n\n", style="neo.think.txt")

    # Live reasoning stream — sliding window of the last _MAX_REASON chars
    if reasoning:
        snippet = reasoning[-_MAX_REASON:]
        # Trim to nearest newline so we don't split mid-sentence
        if len(reasoning) > _MAX_REASON and "\n" in snippet:
            snippet = snippet[snippet.index("\n") + 1:]
        out.append("  ┊ ", style="neo.sep")
        out.append("internal monologue\n", style="dim")
        for line in snippet.splitlines():
            line = line.strip()
            if line:
                out.append("  │  ", style="neo.sep")
                out.append(f"{line[:110]}\n", style="neo.think.txt dim")
        out.append("\n")

    for step in steps:
        stype       = step.get("type", "think")
        icon, style = _STEP_STYLE.get(stype, ("·", "white"))
        label       = step.get("label", "")
        detail      = step.get("detail", "")
        out.append(f"  {icon}  ", style=style)
        out.append(label, style=f"bold {style}")
        if detail:
            out.append("  ›  ", style="dim")
            out.append(f'"{detail}"', style="neo.tool.arg")
        out.append("\n")
    return out


def show_thinking(steps: list) -> None:
    shown: list = []
    with Live(_build_thinking([]), console=console,
              refresh_per_second=15, transient=False) as live:
        for step in steps:
            delay = step.get("_delay", 0.4)
            time.sleep(delay)
            shown.append(step)
            live.update(_build_thinking(shown))
        time.sleep(0.35)
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Panel factories
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_text(raw) -> str:
    """Normalise chunk.content to a plain string.

    The Gemini API (and some other providers) may return content as:
      - str                          → return as-is
      - list[str]                    → join them
      - list[dict]  (content parts)  → extract each part's 'text' key
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(raw) if raw else ""


def _neo_panel(content: str) -> Panel:
    # Guard: content should always be str, but normalise defensively
    if not isinstance(content, str):
        content = _chunk_text(content)
    return Panel(
        Markdown(content) if content.strip() else Text("▋"),
        title="[neo.tag]  Neo  [/neo.tag]",
        border_style="color(57)",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _think_panel(steps: list, reasoning: str = "") -> Panel:
    return Panel(
        _build_thinking(steps, reasoning) if (steps or reasoning) else Text(""),
        border_style="color(99)",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def stream_response(text: str, delay: float = 0.045) -> None:
    words = text.split()
    accumulated = ""
    with Live(_neo_panel("▋"), console=console,
              refresh_per_second=20, transient=False) as live:
        for word in words:
            accumulated += word + " "
            live.update(_neo_panel(accumulated.rstrip()))
            time.sleep(delay)
    console.print()


def show_response(response: str) -> None:
    console.print(_neo_panel(response))
    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Exit
# ─────────────────────────────────────────────────────────────────────────────

def show_exit() -> None:
    console.print()
    console.print(
        Panel(
            "[neo.exit]Session ended.[/neo.exit]\n[neo.hint]Stay curious. -- Neo[/neo.hint]",
            border_style="color(196)",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


@contextmanager
def thinking_spinner(message: str = "Neo is thinking..."):
    with console.status(f"[neo.think.txt]{message}[/neo.think.txt]", spinner="dots"):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Real-mode streaming loop
#
# Phase A — Thinking trace
#   transient=False commits the panel permanently to the terminal when it exits.
#   The trace shows: live reasoning tokens (sliding window) + tool call names.
#
# Phase B — Response streaming
#   A second Live block starts after Phase A exits.
#   Both phases share one generator iterator — Python generators preserve state
#   across context managers, so Phase B continues exactly where Phase A stopped.
# ─────────────────────────────────────────────────────────────────────────────

def _run_real_mode() -> None:
    from langchain_core.messages import AIMessageChunk
    from agents import build_agent
    from tools import rewrite_query

    animate_intro()

    while True:
        try:
            user_input = get_user_input()

            if not user_input:
                continue

            if is_exit_command(user_input):
                show_exit()
                break

            # Phase 1: rewrite query (qwen2.5:1.5b, fast) + build agent
            with console.status(
                "[neo.think.txt]  ◆  Neo is thinking...[/neo.think.txt]",
                spinner="dots",
                spinner_style="color(99)",
            ):
                rewritten = rewrite_query(user_input)
                agent     = build_agent()

            prompt = f"Question: {rewritten}"
            inputs = {"messages": [{"role": "user", "content": prompt}]}

            # Shared iterator — state is preserved across both Live blocks
            stream_iter     = iter(agent.stream(inputs, stream_mode="messages"))
            trace_steps:    list = []
            seen_tool_ids:  set  = set()
            reasoning_text: str  = ""   # accumulated qwen3 thinking tokens
            first_content:  str  = ""   # first answer chunk, bridges A → B
            # Tool-call arg accumulation: args stream in as JSON fragments
            # keyed by the tool's positional index in the tool_calls array.
            tool_idx_to_step: dict = {}  # index → position inside trace_steps
            tool_args_acc:    dict = {}  # index → accumulated raw JSON string

            # ── Phase A: Thinking trace ───────────────────────────────────────
            # Exits (and permanently commits) when first answer token arrives.
            with Live(
                _think_panel(trace_steps, reasoning_text),
                console=console,
                refresh_per_second=20,
                transient=False,
            ) as live_trace:

                for chunk, metadata in stream_iter:

                    # 1. Reasoning tokens — stream into dim monologue window
                    reasoning = ""
                    if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        reasoning_text += reasoning
                        live_trace.update(_think_panel(trace_steps, reasoning_text))

                    # 2. Tool call planning + live arg streaming
                    #
                    # Tool calls arrive as a stream of small chunks:
                    #   - First chunk for a tool: has `id`, `name`, `index`, partial `args`
                    #   - Follow-up chunks:        have only `index` + more `args` fragments
                    # We accumulate args by index, try to parse JSON, and update the
                    # trace step's `detail` live so the user sees the query as it types in.
                    tc_chunks = getattr(chunk, "tool_call_chunks", None) or []
                    need_redraw = False
                    for tc in tc_chunks:
                        tc_id     = (tc.get("id") or "").strip()
                        name      = (tc.get("name") or "").strip()
                        idx       = tc.get("index", 0)
                        args_frag = tc.get("args", "") or ""

                        # — New tool call: register it
                        if tc_id and tc_id not in seen_tool_ids and name:
                            seen_tool_ids.add(tc_id)
                            tool_args_acc[idx]    = args_frag
                            tool_idx_to_step[idx] = len(trace_steps)
                            trace_steps.append({"type": "tool", "label": name, "detail": ""})
                            need_redraw = True

                        # — Streaming args fragment: accumulate and try to parse
                        elif args_frag and idx in tool_args_acc:
                            tool_args_acc[idx] += args_frag
                            raw = tool_args_acc[idx]
                            # Attempt JSON parse — gives clean value when complete
                            detail = ""
                            try:
                                parsed = json.loads(raw)
                                # Extract the first (usually only) argument value
                                detail = str(next(iter(parsed.values()), ""))
                            except (json.JSONDecodeError, StopIteration):
                                # JSON not complete yet — show a clean partial preview
                                detail = raw.lstrip('{" ').rstrip('"')
                            detail = detail[:80].strip()
                            step_pos = tool_idx_to_step.get(idx)
                            if step_pos is not None and detail:
                                trace_steps[step_pos]["detail"] = detail
                                need_redraw = True

                    if need_redraw:
                        live_trace.update(_think_panel(trace_steps, reasoning_text))

                    # 3. First answer chunk → finalise trace, hand off to Phase B
                    # _chunk_text normalises list-of-parts (Gemini/Gemma) → plain str
                    content = _chunk_text(getattr(chunk, "content", "") or "")
                    if content and isinstance(chunk, AIMessageChunk) and not tc_chunks:
                        trace_steps.append({"type": "done", "label": "Synthesising answer..."})
                        live_trace.update(_think_panel(trace_steps, reasoning_text))
                        time.sleep(0.15)
                        first_content = content
                        break

            # Trace panel permanently committed to terminal ↑
            console.print()

            # ── Phase B: Response streaming ───────────────────────────────────
            # stream_iter continues from where Phase A stopped
            accumulated = first_content
            with Live(
                _neo_panel(accumulated or "▋"),
                console=console,
                refresh_per_second=20,
                transient=False,
            ) as live_resp:

                if accumulated:
                    live_resp.update(_neo_panel(accumulated))

                for chunk, metadata in stream_iter:
                    content = _chunk_text(getattr(chunk, "content", "") or "")
                    if content and isinstance(chunk, AIMessageChunk):
                        accumulated += content
                        live_resp.update(_neo_panel(accumulated))

            console.print()

        except KeyboardInterrupt:
            show_exit()
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    _USE_MOCK = "--mock" in sys.argv

    if _USE_MOCK:
        class _MockScenario(TypedDict):
            steps: list[_MockStep]
            response: str

        _MOCK_SCENARIOS: list[_MockScenario] = [
            {
                "steps": [
                    {"type": "think", "label": "Analyzing query...",                               "_delay": 0.35},
                    {"type": "tool",  "label": "bm25_search_tool",      "detail": "node version",  "_delay": 0.75},
                    {"type": "tool",  "label": "similarity_search_tool", "detail": "node on PATH",  "_delay": 0.85},
                    {"type": "done",  "label": "Synthesizing from 5 results...",                    "_delay": 0.45},
                ],
                "response": (
                    "**Node.js** is installed on your system.\n\n"
                    "| Field   | Value |\n"
                    "|---------|-------|\n"
                    "| Version | `18.17.0` |\n"
                    "| Path    | `/usr/bin/node` |\n"
                    "| Source  | APT |\n\n"
                    "The binary is on your `$PATH`."
                ),
            },
            {
                "steps": [
                    {"type": "think", "label": "Query needs hardware context.",                     "_delay": 0.30},
                    {"type": "tool",  "label": "similarity_search_tool", "detail": "RAM memory",   "_delay": 0.80},
                    {"type": "done",  "label": "Found hardware snapshot.",                          "_delay": 0.35},
                ],
                "response": (
                    "Your system has **16 GB RAM**.\n\n"
                    "- Used: `4.2 GB`\n"
                    "- Free: `11.8 GB`\n"
                    "- Swap: `2 GB` (0 % used)"
                ),
            },
            {
                "steps": [
                    {"type": "think", "label": "Checking all local indexes...",                    "_delay": 0.40},
                    {"type": "tool",  "label": "bm25_search_tool",      "detail": "config",        "_delay": 0.65},
                    {"type": "tool",  "label": "similarity_search_tool", "detail": "config files",  "_delay": 0.70},
                    {"type": "tool",  "label": "web_search_tool",        "detail": "Linux config",  "_delay": 0.95},
                    {"type": "error", "label": "No relevant results found.",                        "_delay": 0.40},
                ],
                "response": "I could not find any information about that in your local system data or via web search.",
            },
        ]
        _idx = 0

        animate_intro()

        while True:
            try:
                user_input = get_user_input()
                if not user_input:
                    continue
                if is_exit_command(user_input):
                    show_exit()
                    break
                scenario = _MOCK_SCENARIOS[_idx % len(_MOCK_SCENARIOS)]
                _idx += 1
                show_thinking(copy.deepcopy(scenario["steps"]))
                stream_response(scenario["response"])
            except KeyboardInterrupt:
                show_exit()
                break

    else:
        _run_real_mode()
