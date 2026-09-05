"""
app.py — The GUI. Run this file.

    uv run python app.py

Layout:

    +--------------------------+-----------------------------------+
    |                          | [Context] [Memory] [Tools] [Trace] |
    |   chat with the agent    |                                   |
    |                          |   live panels showing what the    |
    |                          |   agent can actually SEE          |
    |  [ type here ]   [send]  |                                   |
    +--------------------------+-----------------------------------+

The panels are the point. The chat is just how you poke it.

Note the deliberate split: the chat window on the left keeps the FULL history
for you to read, while the Context panel on the right shows only the messages
still being sent to the model. When those two disagree, you are looking at the
context window limit with your own eyes.
"""

import gradio as gr
import ollama

from teacher_assistant import settings
from teacher_assistant.agents import subagent
from teacher_assistant.agents.main import run_turn
from teacher_assistant.mcp.client import MCPClient
from teacher_assistant.memory import store
from teacher_assistant.skills import loader

# ---------------------------------------------------------------------------
# STARTUP
# ---------------------------------------------------------------------------
print("Connecting to MCP server and web research...")
MCP = MCPClient(settings.MCP_SERVER_PATH)
MCP.connect()
print(f"  connected: {MCP.server_names}")
print(f"  tools available: {[t.name for t in MCP.tools]}")


def _warm_up():
    """Load both models before the UI opens so the first turn stays responsive."""
    try:
        ollama.embed(
            model=settings.EMBED_MODEL,
            input="hi",
            options=settings.chat_options(),
            keep_alive=settings.KEEP_ALIVE,
        )
        print(f"  {settings.EMBED_MODEL} warmed up and resident.")
    except Exception as e:
        print(f"  WARNING: could not warm {settings.EMBED_MODEL} ({e}).")

    try:
        ollama.chat(
            model=settings.MODEL,
            messages=[{"role": "user", "content": "hi"}],
            options=settings.chat_options(
                temperature=settings.TEMPERATURE,
                num_predict=1,
            ),
            keep_alive=settings.KEEP_ALIVE,
        )
        print(f"  {settings.MODEL} warmed up and resident.")
    except Exception as e:
        print(f"  WARNING: could not reach Ollama ({e}). Run `ollama serve`.")


_warm_up()


# ---------------------------------------------------------------------------
# PANEL RENDERERS — turn state into readable markdown
# ---------------------------------------------------------------------------
def render_context(messages: list[dict]) -> str:
    """SHORT-TERM MEMORY: exactly what gets sent to the model."""
    limit = settings.SHORT_TERM_MAX_TURNS * 2
    header = (
        f"### Short-term memory\n"
        f"`{len(messages)} / {limit}` messages in the window "
        f"(= {settings.SHORT_TERM_MAX_TURNS} turns, set in "
        "`teacher_assistant/settings.py`)\n\n"
    )
    if len(messages) >= limit:
        header += "> **FULL.** Every new message now pushes an old one out permanently.\n\n"
    if not messages:
        return header + "_empty — nothing has been said yet_"

    rows = []
    for m in messages:
        text = m["content"].replace("\n", " ")
        text = text[:90] + ("..." if len(text) > 90 else "")
        rows.append(f"| {m['role']} | {text} |")
    return header + "| role | content |\n|---|---|\n" + "\n".join(rows)


def render_memory() -> str:
    """LONG-TERM MEMORY: the contents of memories.json."""
    memories = store.load()
    memory_label = settings.MEMORY_PATH.relative_to(settings.PROJECT_ROOT).as_posix()
    header = f"### Long-term memory\n`{len(memories)}` fact(s) in `{memory_label}`\n\n"
    if not memories:
        return header + "_empty — tell the agent something the gradebook doesn't know (an accommodation, a circumstance, a preference)_"
    return header + "\n".join(f"- **#{m['id']}** {m['text']}" for m in memories)


def render_tools() -> str:
    """Show MCP-discovered and client-owned tools available to the agent."""
    out = ["### Tools available to the agent"]
    server_label = settings.MCP_SERVER_PATH.relative_to(settings.PROJECT_ROOT).as_posix()
    out.append(
        f"_Sources: `course-tools` MCP server ({server_label}) and "
        "client-owned key-free web research._\n"
    )
    for tool in MCP.tools:
        params = ", ".join(tool.input_schema.get("properties", {}).keys()) or ""
        source = MCP.tool_sources.get(tool.name, "MCP")
        out.append(
            f"- **`{tool.name}({params})`** _[{source}]_ — "
            f"{tool.description.strip().splitlines()[0]}"
        )
    out.append("- **`load_skill(name)`** — _client-side, not from MCP_")

    owned = subagent.owned_skills()
    out.append("\n### Skills the main agent can load")
    out.append("_Full instructions load on demand. That's progressive disclosure._\n")
    for skill in loader.list_skills():
        if skill["name"].lower() not in owned:
            out.append(f"- **{skill['name']}** — {skill['description']}")

    out.append("\n### Specialists it can delegate to")
    out.append("_Each runs its own agent loop in its own context. See the Subagent tab._\n")
    for spec in subagent.list_subagents():
        out.append(
            f"- **{spec['name']}** — {spec['description']}\n"
            f"  - tools: `{'`, `'.join(spec['tools'])}` _(and nothing else)_\n"
            f"  - owns skill: **{spec['skill']}** _(hidden from the main agent "
            f"while 'Delegate to subagent' is on)_"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# THE SUBAGENT PANEL — two context windows, side by side
# ---------------------------------------------------------------------------
# The most common misunderstanding about subagents is that they are "just more
# prompting". This panel is the rebuttal: it shows a SECOND context window that
# the main agent never sees, fills up, and then throws away.
EMPTY_SUBAGENT = (
    "### Subagent\n_No subagent has run yet._\n\n"
    "Try: **`draft a check-in email to Marcus`**\n\n"
    "The main agent will hand the whole job to `email-writer`, which runs its "
    "own loop in its own context and returns only the finished draft.\n\n"
    "Then: **Reset conversation** → untick **Delegate to subagent** → ask again. "
    "Compare the grades in the two emails against the Tools panel.\n\n"
    "> The reset is not optional. Skip it and the old email is still in the "
    "conversation, so the main agent just copies it and the comparison shows "
    "you nothing."
)

# Gradio event handlers are stateless, so the last run lives here. One user, one
# demo -- a module-level variable is the honest, readable choice.
_LAST_SUBAGENT = None


def render_subagent(panel: dict | None) -> str:
    if not panel:
        return EMPTY_SUBAGENT

    running = panel["status"] == "running"
    out = [
        f"### Subagent `{panel['name']}` — "
        + ("**running…**" if running else "**finished**"),
        f"\n**Task it was given:** _{panel['task']}_\n",
        "#### What it can see (its ENTIRE world)",
        f"- its own persona — _not the main agent's_",
        f"- tools: `{'`, `'.join(panel['tools'])}` — **not** the other "
        f"{len(MCP.tools) - len(panel['tools'])} the main agent has",
        f"- skill: **{panel['skill']}** (~{panel['skill_tokens']} tokens)",
    ]
    if panel["briefing"]:
        out.append(f"- briefing passed down by the parent ({len(panel['briefing'])}):")
        out += [f"    - _{b}_" for b in panel["briefing"]]
    else:
        out.append("- _no briefing_ — it knows nothing about the conversation")

    out.append(
        "\n#### What it CANNOT see\n"
        "the chat history · long-term memory · the other tools · the teacher's "
        "actual words. It starts empty. Everything above had to be handed to it."
    )

    out.append("\n#### Its loop")
    out += [f"- {s}" for s in panel["steps"]] or ["- _starting…_"]

    if not running:
        returned = len(panel["result"]) // 4
        spent = max(panel["prompt_tokens"] - returned, 0)
        out.append(
            f"\n#### The trade\n"
            f"| | tokens |\n|---|---|\n"
            f"| context the subagent burned | **{panel['prompt_tokens']}** |\n"
            f"| what it handed to the main agent | **~{returned}** |\n"
            f"| spent in a context that no longer exists | **~{spent}** |\n"
        )
        out.append(
            f"~{spent} tokens of skill text and gradebook data were read, used, "
            "and thrown away. They are not in the Context tab and never will be. "
            "The main agent got the finished work without paying to store how it "
            "was made."
        )
        # ---- MODIFY HERE ----
        # Be honest with your class about the scale here. At FOUR assignments and
        # ONE student, the parent's total prompt is about the same either way --
        # flip the toggle and compare, the numbers barely move. The saving only
        # compounds when the delegated job is big or repeated (twelve students,
        # a long document, a ten-step procedure).
        #
        # The win you CAN see at this scale is reliability, and it is the better
        # lesson anyway: the specialist's persona says "call student_report
        # BEFORE you write a single word", and student_report is the only tool it
        # has. Narrow job + narrow menu = it does the right thing. Untick the
        # toggle and the main agent, holding many tools and a whole conversation,
        # will often skip the lookup and invent the grades outright.
        out.append(
            "\n> **Try it both ways:** **Reset conversation**, untick the toggle, "
            "ask again, then check the grades against the Tools panel. A general "
            "agent juggling many tools and a conversation often loads the skill, "
            "decides it doesn't need the lookup, and invents the numbers. A "
            "specialist with one job and one tool does not.\n>\n"
            "> Reset first, always — otherwise the old email is still in the "
            "conversation and the main agent simply copies it."
        )
        out.append("\n#### What it handed back\n```\n" + panel["result"] + "\n```")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# THE MAIN EVENT HANDLER
# ---------------------------------------------------------------------------
def on_send(user_text, chat, messages, retrieval_mode, use_subagent):
    """Runs one turn and streams every intermediate state into the GUI."""
    global _LAST_SUBAGENT

    if not user_text.strip():
        yield tuple(gr.skip() for _ in range(8))
        return

    chat = chat + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]
    trace_lines = []

    # Paint the submitted message before retrieval or any model call starts.
    # The status beneath the chat stays visible even when the Trace tab is not.
    yield (
        "",
        chat,
        gr.skip(),
        "### Trace\n**Working...**",
        gr.skip(),
        gr.skip(),
        "**Working...**",
        gr.skip(),
    )

    # run_turn is a generator; each event updates a different part of the UI.
    for kind, payload in run_turn(MCP, messages, user_text, retrieval_mode, use_subagent):
        if kind == "trace":
            trace_lines.append(payload)
            yield (
                gr.skip(), gr.skip(), gr.skip(),
                "### Trace\n" + "\n\n".join(trace_lines),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
        elif kind == "chart":
            # A tool rendered a PNG. Show the image itself in the chat, just
            # above the answer that's about to stream in. The LLM never sees
            # this image -- tools return artifacts, the model returns words.
            chat.insert(len(chat) - 1, {"role": "assistant", "content": gr.Image(payload)})
            yield (
                gr.skip(), chat, gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
        elif kind == "subagent":
            # A second agent is running. Its state goes to its own tab -- never
            # into the chat, and never into `messages`. That's the isolation.
            _LAST_SUBAGENT = payload
            yield (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), render_subagent(_LAST_SUBAGENT),
            )
        elif kind == "token":
            chat[-1]["content"] += payload
            yield (
                gr.skip(), chat, gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            )
        elif kind == "stats":
            stats = (
                f"**Prompt size:** {payload['prompt_tokens']} tokens &nbsp;|&nbsp; "
                f"**Skills loaded:** {payload['skills_loaded']} &nbsp;|&nbsp; "
                f"**Delegations:** {payload['delegations']}"
            )
            yield (
                gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                gr.skip(), gr.skip(), stats, gr.skip(),
            )
        elif kind == "done":
            messages = payload
            # Reflection has completed. Refresh persisted memory and the bounded
            # model context exactly once at the end of the turn.
            yield (
                gr.skip(), gr.skip(), messages, gr.skip(),
                render_context(messages), render_memory(), gr.skip(), gr.skip(),
            )


# ---------------------------------------------------------------------------
# DEMO BUTTONS — so you can re-run a beat without retyping
# ---------------------------------------------------------------------------
def reset_conversation():
    """Clear short-term memory only. Long-term facts survive — that's the point."""
    global _LAST_SUBAGENT
    _LAST_SUBAGENT = None
    return [], [], render_context([]), render_memory(), "", "", EMPTY_SUBAGENT


def wipe_long_term():
    store.wipe()
    return render_memory()


def fill_context(messages):
    """Stuff the window with filler so you can show overflow instantly."""
    filler = []
    for i in range(settings.SHORT_TERM_MAX_TURNS):
        filler.append({"role": "user", "content": f"[filler question #{i + 1}]"})
        filler.append({"role": "assistant", "content": f"[filler answer #{i + 1}]"})
    messages = (messages + filler)[-settings.SHORT_TERM_MAX_TURNS * 2 :]
    return messages, render_context(messages)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Local Agent Demo") as demo:
    gr.Markdown(
        f"# Teacher's Assistant — Local Agent Demo &nbsp;·&nbsp; `{settings.MODEL}`\n"
        "An agent for a professor: gradebook tools over **MCP** · **short-term** vs "
        "**long-term** memory · **skills** with progressive disclosure · a "
        "**subagent** it delegates whole jobs to · key-free **web research** "
        "for outside knowledge."
    )

    # Short-term memory really is just a Python list living in this State.
    messages_state = gr.State([])

    with gr.Row():
        # ---------------- LEFT: chat ----------------
        with gr.Column(scale=5):
            # Gradio 6 chat messages are {"role": ..., "content": ...} dicts --
            # the same shape the Ollama API uses, so no conversion needed.
            chatbot = gr.Chatbot(height=460, show_label=False)
            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask about a student or the class, request a chart, or share context only you'd know...",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            stats_md = gr.Markdown("")

            with gr.Row():
                retrieval_mode = gr.Radio(
                    ["semantic", "keyword"],
                    value="semantic",
                    label="Memory retrieval strategy",
                    info="Ask the same question with each. Watch the Trace panel.",
                    scale=2,
                )
                with gr.Column(scale=1):
                    # ---- THE A/B SWITCH ----
                    # Ask "draft a check-in email to Marcus" with this ON, then
                    # again with it OFF. Same email either way -- but watch the
                    # "Prompt size" number, and the Subagent tab.
                    use_subagent = gr.Checkbox(
                        value=True,
                        label="Delegate to subagent",
                        info="Off = the main agent does the whole job itself.",
                    )
                    reset_btn = gr.Button("Reset conversation", size="sm")
                    fill_btn = gr.Button("Fill context window", size="sm")
                    wipe_btn = gr.Button("Wipe long-term memory", size="sm", variant="stop")

        # ---------------- RIGHT: the panels that teach ----------------
        with gr.Column(scale=4):
            with gr.Tab("Context"):
                context_md = gr.Markdown(render_context([]))
            with gr.Tab("Memory"):
                memory_md = gr.Markdown(render_memory())
            with gr.Tab("Tools"):
                gr.Markdown(render_tools())
            with gr.Tab("Subagent"):
                subagent_md = gr.Markdown(EMPTY_SUBAGENT)
            with gr.Tab("Trace"):
                trace_md = gr.Markdown("### Trace\n_Send a message to see the agent's loop._")

    # ---- wiring ----
    outputs = [
        msg_box, chatbot, messages_state, trace_md, context_md, memory_md,
        stats_md, subagent_md,
    ]
    inputs = [msg_box, chatbot, messages_state, retrieval_mode, use_subagent]

    send_btn.click(on_send, inputs, outputs, stream_every=0.05)
    msg_box.submit(on_send, inputs, outputs, stream_every=0.05)

    reset_btn.click(
        reset_conversation,
        None,
        [chatbot, messages_state, context_md, memory_md, trace_md, stats_md, subagent_md],
    )
    wipe_btn.click(wipe_long_term, None, memory_md)
    fill_btn.click(fill_context, messages_state, [messages_state, context_md])


if __name__ == "__main__":
    # inbrowser=True pops the tab automatically. share=True would give you a
    # public link (needs internet) -- not needed, everything runs locally.
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
