"""
subagent.py — SUBAGENTS.

Read main.py first. This file is the SAME LOOP, smaller, and that is the point:
a subagent is not a new kind of object. It is another agent loop, with its own
prompt, its own tools, and -- the part that matters -- ITS OWN CONTEXT WINDOW.

WHY WOULD YOU EVER WANT THAT?
-----------------------------
Drafting a student email needs three bulky things in context:

    the check-in-email skill body   ~600 tokens
    that student's full record      ~200 tokens
    the rules for how to write it   (also in the skill)

...and it produces ONE small thing: a 120-word email.

If the main agent does that itself, all of it lands in the main agent's context
and STAYS there for the rest of the conversation, crowding out the gradebook,
the memories, and the chat. If a subagent does it, the parent sees only the
finished email. The bulky middle is thrown away.

    Big input, small output, no need to remember the middle.
    That is the shape of a job worth delegating. It is the ONLY test.

THE OTHER HALF: A SUBAGENT STARTS EMPTY
---------------------------------------
It does not inherit the conversation, the memories, or anything the teacher
said. Whatever it needs, the parent must PASS IN -- see `briefing` below.
Students always expect delegation to be free. It isn't: you pay for it in
briefing, and a bad briefing produces a confident, isolated, wrong answer.

WHY IS THIS A SEPARATE FILE INSTEAD OF REUSING main.py's FUNCTIONS?
--------------------------------------------------------------------
So you can read it start to finish on its own and compare the two side by side.
Yes, `_decide` below is nearly a copy of `decide` in main.py. That duplication
is deliberate teaching, not an accident -- put the two files next to each other
and notice they are the same four steps.
"""

import json

from teacher_assistant import llm, settings
from teacher_assistant.skills import loader


# ---------------------------------------------------------------------------
# THE ROSTER OF SPECIALISTS
# ---- MODIFY HERE ----
# Adding a subagent is adding a dict to this list. It needs four things: who it
# is (persona), what it may touch (tools), what procedure it follows (skill),
# and how the parent should describe it to itself (description).
#
# Note `tools`: this subagent can call student_report and NOTHING ELSE. It
# cannot chart, cannot list the roster, cannot see deadlines. Narrowing the tool
# list is most of what makes a specialist reliable -- fewer choices, fewer wrong
# choices. In production it is also a real security boundary.
# ---------------------------------------------------------------------------
SUBAGENTS = [
    {
        "name": "email-writer",
        "description": (
            "Drafts a check-in email from the teacher to ONE student. Hand it "
            "the student's name and the reason for reaching out; it looks up "
            "that student's record itself and returns a finished draft."
        ),
        "tools": ["student_report"],
        "skill": "check-in-email",
        "persona": """You draft short check-in emails FROM a university professor
TO one of their students. This one job is all you do.

You do NOT know any grades, attendance, or late submissions. That data lives in
tools. Call student_report for the student named in your task BEFORE you write a
single word, and copy every number from it exactly, digit for digit.

Your entire output is the email itself. No preamble, no "here is the draft", no
commentary afterwards. Just the email.""",
    },
]


def list_subagents() -> list[dict]:
    return SUBAGENTS


def find(name: str) -> dict | None:
    for spec in SUBAGENTS:
        if spec["name"].lower() == str(name).strip().lower():
            return spec
    return None


def owned_skills() -> set[str]:
    """Skills that belong to a subagent, so the PARENT is not offered them.

    A design decision worth saying out loud in class: check-in-email used to be
    the parent's skill. Now the specialist owns it and the parent is never shown
    it. Two things that can do the same job is exactly how a small model ends up
    dithering between them. Give a capability precisely one home.
    """
    return {s["skill"].lower() for s in SUBAGENTS}


# ---------------------------------------------------------------------------
# STEP A: DECIDE — identical idea to agent.decide(), narrower menu
# ---------------------------------------------------------------------------
def _build_schema(tools) -> dict:
    """Constrain the subagent to ITS tools only. Same enum trick as the parent:
    a tool name outside this list is not 'discouraged', it is ungeneratable."""
    arg_properties = {}
    for tool in tools:
        for arg_name, spec in tool.input_schema.get("properties", {}).items():
            arg_properties[arg_name] = {"type": spec.get("type", "string")}

    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "tool": {"type": "string", "enum": [t.name for t in tools] + ["none"]},
            "args": {
                "type": "object",
                "properties": arg_properties,
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "tool", "args"],
    }


def _decide(system_prompt: str, task: str, schema: dict) -> dict:
    response = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"YOUR TASK: {task}\n\n---\n"
                "Decide your NEXT action. Look up the student's record if you do "
                'not have it yet; otherwise pick "none" and write the email.',
            },
        ],
        response_schema=schema,
        schema_name="subagent_decision",
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_DECISION_TOKENS,
    )
    try:
        decision = json.loads(response.content)
    except json.JSONDecodeError:
        return {"reasoning": "unparseable", "tool": "none", "args": {}}
    if not isinstance(decision.get("args"), dict):
        decision["args"] = {}
    return decision


def _repair_args(system_prompt: str, task: str, tool) -> dict:
    """Same repair trick as the parent: re-ask using the tool's OWN schema."""
    response = llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"YOUR TASK: {task}\n\n---\nGive the arguments for "
                f"calling `{tool.name}`.",
            },
        ],
        response_schema=tool.input_schema,
        schema_name=f"{tool.name}_arguments",
        temperature=settings.TEMPERATURE,
        max_tokens=settings.MAX_DECISION_TOKENS,
    )
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# PROMPT ASSEMBLY — the subagent's ENTIRE world
# ---------------------------------------------------------------------------
def _build_system_prompt(
    spec, skill_body, briefing, observations, include_tools: bool = True,
) -> str:
    """Compare this to agent.build_system_prompt(). Notice what is MISSING:

        no conversation history      no long-term memory
        no other skills              no other tools

    That absence is the feature. The subagent cannot be distracted by, or
    contradict, a conversation it was never shown. It also cannot use anything
    from that conversation -- which is exactly why `briefing` has to exist.
    """
    parts = [spec["persona"]]

    if briefing:
        lines = ["\n## WHAT THE PROFESSOR'S ASSISTANT PASSED YOU"]
        lines += [f"- {b}" for b in briefing]
        lines.append(
            "Let this shape the tone of the email -- gentler, no piling on -- but "
            "NEVER state in the email that the professor discussed this with anyone."
        )
        parts.append("\n".join(lines))

    if include_tools:
        lines = ["\n## TOOLS YOU CAN CALL"]
        for tool in spec["_tools"]:
            params = ", ".join(tool.input_schema.get("properties", {}).keys()) or "none"
            description = "\n".join(
                f"  {line.strip()}"
                for line in tool.description.strip().splitlines()
            )
            lines.append(f"- {tool.name}({params}):\n{description}")
        parts.append("\n".join(lines))

    # The bulky part. It lives HERE and only here.
    parts.append(f"\n## YOUR PROCEDURE — follow it exactly\n\n{skill_body}")

    if observations:
        lines = ["\n## TOOL RESULTS"]
        lines += [f"### {name}\n{result}" for name, result in observations]
        lines.append("These are authoritative. Use these exact numbers; invent nothing.")
        parts.append("\n".join(lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# THE SUBAGENT LOOP
# ---------------------------------------------------------------------------
def run(mcp, spec: dict, task: str, briefing: list[str]):
    """Run one subagent to completion. A GENERATOR, like run_turn(), so the GUI
    can watch a second agent think in real time.

    Yields (kind, payload):
        ("trace", str)   -> a line for the parent's Trace panel (indented there)
        ("panel", dict)  -> the Subagent tab's live state
        ("token", str)   -> a piece of the finished work for the chat stream
        ("result", str)  -> the finished work, handed back to the parent
    """
    spec = dict(spec)
    spec["_tools"] = [t for t in mcp.tools if t.name in spec["tools"]]

    panel = {
        "name": spec["name"],
        "status": "running",
        "task": task,
        "briefing": list(briefing),
        "tools": spec["tools"],
        "skill": spec["skill"],
        "skill_tokens": 0,
        "steps": [],
        "result": "",
        "prompt_tokens": 0,
    }

    yield ("trace", f"**SUBAGENT `{spec['name']}` started** — brand new, empty context")
    yield ("panel", dict(panel))

    # --- load its procedure -------------------------------------------------
    # Unconditional, on purpose. The parent already decided this job needs this
    # specialist; a specialist that might not read its own instructions is just a
    # coin flip. The parent still uses on-demand load_skill (progressive
    # disclosure); a subagent's skill IS its job description.
    skill_body = loader.load_skill(spec["skill"])
    panel["skill_tokens"] = len(skill_body) // 4
    line = (
        f"loaded skill `{spec['skill']}` (+~{panel['skill_tokens']} tokens) "
        "— into THIS context, never the parent's"
    )
    panel["steps"].append(line)
    yield ("trace", line)
    yield ("panel", dict(panel))

    # --- its own DECIDE -> ACT loop ----------------------------------------
    schema = _build_schema(spec["_tools"])
    observations: list[tuple[str, str]] = []
    already_called: set[str] = set()

    for step in range(settings.SUBAGENT_MAX_TOOL_STEPS):
        system_prompt = _build_system_prompt(spec, skill_body, briefing, observations)
        decision = _decide(system_prompt, task, schema)
        tool = decision["tool"]

        if tool == "none":
            line = f"step {step + 1}: has what it needs, writing"
            panel["steps"].append(line)
            yield ("trace", line)
            yield ("panel", dict(panel))
            break

        tool_obj = next((t for t in spec["_tools"] if t.name == tool), None)
        if tool_obj is not None:
            required = tool_obj.input_schema.get("required", [])
            if any(arg not in decision["args"] for arg in required):
                decision["args"] = _repair_args(system_prompt, task, tool_obj)

        signature = tool + json.dumps(decision["args"], sort_keys=True)
        if signature in already_called:
            break
        already_called.add(signature)

        line = f"step {step + 1}: calling `{tool}({json.dumps(decision['args'])})` via MCP"
        panel["steps"].append(line)
        yield ("trace", line)
        yield ("panel", dict(panel))

        try:
            result = mcp.call_tool(tool, decision["args"])
        except Exception as e:
            result = f"Tool failed: {e}"
        observations.append((tool, result))

        line = f"&nbsp;&nbsp;&nbsp;&nbsp;-> {result[:160]}"
        panel["steps"].append(line)
        yield ("trace", line)
        yield ("panel", dict(panel))

    # --- ANSWER -------------------------------------------------------------
    # This is finished, teacher-facing work, so stream it directly through the
    # parent instead of making the parent regenerate the same text.
    system_prompt = _build_system_prompt(
        spec, skill_body, briefing, observations, include_tools=False,
    )
    stream = llm.stream_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"YOUR TASK: {task}\n\n---\nWrite the email now. Output "
                "the email and nothing else.",
            },
        ],
        max_tokens=settings.MAX_ANSWER_TOKENS,
    )
    output = ""
    for chunk in stream:
        piece = chunk.content
        output += piece
        if piece:
            yield ("token", piece)
        if chunk.prompt_tokens:
            panel["prompt_tokens"] = chunk.prompt_tokens
    output = output.strip()

    panel["result"] = output
    panel["status"] = "done"

    yield (
        "trace",
        f"**SUBAGENT `{spec['name']}` finished** — returned {len(output.split())} words. "
        f"Its {panel['prompt_tokens']}-token context is now DISCARDED.",
    )
    yield ("panel", dict(panel))
    yield ("result", output)
