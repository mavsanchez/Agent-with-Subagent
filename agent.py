"""
agent.py — THE AGENT LOOP. Read this file first.

Everything else in this project is a supporting character. This is the agent.

An "agent" is not a magic object. It is a while-loop that:

    1. shows an LLM a list of tools
    2. asks "do you want to call one?"
    3. if yes -> runs it, pastes the result back into the prompt, goes to 1
    4. if no  -> writes the final answer

That's it. That is the whole idea. Frameworks like LangGraph wrap this loop in
abstractions; here it's spelled out so you can watch it run.

WHY TWO LLM CALLS PER STEP?
---------------------------
gemma3:4b was never trained for function calling -- Ollama won't even accept a
`tools=` parameter for it. So we split the job:

    DECIDE  -> constrained JSON output. Tiny, fast, deterministic. Cannot
               hallucinate a tool name, because the schema's `enum` makes
               invalid names literally ungeneratable.
    ANSWER  -> normal streaming text, with tool results in context.

This is more code than `tools=[...]`, but it works on ANY model, and you can
see exactly what a "tool call" really is: a string the model wrote, that we
parsed. There is no magic underneath the frameworks either.
"""

import json

import ollama

import config
import memory
import skills_loader
import subagent

# ---------------------------------------------------------------------------
# The agent's personality and rules.
# ---- MODIFY HERE ---- This is the highest-leverage text in the project.
# Change one sentence and the agent's whole behavior changes.
# ---------------------------------------------------------------------------
PERSONA = """You are the private assistant of a university professor. You help
them understand how their class and individual students are doing. Be concise,
professional, and concrete. Every "you" in the conversation is the teacher.

Critical rules:
- You do NOT know any grades, attendance, statistics, or deadlines. That data
  lives in tools. Never guess a score, a name, or a date -- call a tool.
- You are bad at arithmetic. Use the calculate tool for any NEW math. But
  math that already has an answer earlier in the conversation is DONE --
  never recompute it, re-verify it, or repeat its answer in a later reply.
- Copy numbers from tool results EXACTLY, digit for digit. Never reformat them,
  never add commas, never round further. If the tool says 8839880, you say 8839880.
- Numbers are the start of your job, not the end. When you report data, say
  what it MEANS: name the trend, the outlier, the thing worth acting on.
- Never mention tool names, skills, or "loading" anything in your reply, and
  never write out a tool call. The teacher wants the answer, not your plumbing.
- Only state a fact from a tool if it actually appears under TOOL RESULTS
  below. If it isn't there, you did not look it up -- so do not claim you did.
- Never invent a REASON, a date, or a deadline. If the WHAT YOU REMEMBER
  section gives a cause, use that exact cause. If it doesn't, plainly say the
  reason isn't something you have -- in your own words -- and stop. Never
  substitute a plausible-sounding explanation, and never say you "checked"
  anything that isn't in TOOL RESULTS below.
- Answer in 2-4 sentences unless the teacher asks for detail."""

# Appended only to the FINAL answer call. By this point the loop is over, so the
# agent must commit to an answer instead of narrating more plans.
ANSWER_INSTRUCTION = """
Write your reply to the teacher NOW.
Answer ONLY their most recent message -- earlier questions in the conversation
were already answered, never answer them again. You have already gathered
everything you are going to gather. Never ask the teacher to run a tool or
call a function -- you are the one with the tools, not them. Do not describe
what you are about to do, and never output code blocks, JSON, or tool syntax
like tool_name(...) -- plain prose only. If the teacher was just sharing
information rather than asking a question, acknowledge it in one sentence and
stop.

If the WHAT YOU REMEMBER section above holds anything relevant to this
question, it MUST shape your answer -- say what the numbers mean in light of
it, in the same breath as reporting them. Numbers alone are a spreadsheet;
the teacher is asking you because you know the context too. Just answer."""


# ---------------------------------------------------------------------------
# SHORT-TERM MEMORY: it really is just a list.
# ---------------------------------------------------------------------------
def trim_short_term(messages: list[dict]) -> list[dict]:
    """Drop the oldest messages once we exceed the window.

    THIS FUNCTION IS THE DEMO. When a message falls off the front of this list,
    the agent does not "sort of remember" it. The bytes are never sent to the
    model. It is gone. That is what a context window limit actually is, and
    it's why the next section of this project exists.
    """
    max_messages = config.SHORT_TERM_MAX_TURNS * 2  # one user + one assistant
    return messages[-max_messages:]


# ---------------------------------------------------------------------------
# WHICH SKILLS ARE EVEN ON THE TABLE?
# ---------------------------------------------------------------------------
# ---- MODIFY HERE ----
# A skill is a PROCEDURE -- it answers "do this for me", never "here's some news".
# But "Marcus emailed me about his concussion" and "draft an email to Marcus"
# look almost identical to a 4B model, and it would cheerfully load the
# email-writing skill and produce an email nobody asked for.
#
# We tried fixing that in the skill's own description ("never load this just
# because an email is mentioned"). It did not hold. So we gate it in Python:
# unless the message actually asks for something to be produced, the skills
# never enter the prompt and `load_skill` is not even a choosable option.
#
# This IS a blunt keyword check, and that's a real tradeoff worth naming in
# class: ask for "a note to Marcus" and it won't fire. A production system
# would use a small classifier or embed the message against each skill
# description (the same trick memory.py uses for recall). The lesson is the
# pattern, not the word list: when a rule really matters, enforce it in code.
REQUEST_WORDS = (
    "draft", "write", "compose", "report", "summary", "summarize",
    "recap", "prepare", "put together",
)


def skills_on_offer(user_text: str, skills: list[dict]) -> list[dict]:
    """Only offer skills when the teacher is actually asking for something."""
    return skills if any(w in user_text.lower() for w in REQUEST_WORDS) else []


# ---------------------------------------------------------------------------
# PROMPT ASSEMBLY — everything the model gets to see
# ---------------------------------------------------------------------------
def build_system_prompt(
    tools, memories, skills, loaded_skills, observations, history=(),
    subagents=(), delegated=(),
) -> str:
    """Glue the agent's whole world into one string.

    Worth noticing: this is rebuilt from scratch on EVERY step. Prompts are not
    stateful. Anything you want the model to know, you paste in, every time.
    """
    parts = [PERSONA]

    # --- the conversation so far, as LABELLED HISTORY, not as live messages ---
    # ---- MODIFY HERE ----
    # This is the fix for a failure worth demoing. Old user messages used to be
    # sent as real `user` messages, so the model saw four unanswered questions
    # and re-answered the oldest one every turn ("The product of 234 and 432
    # is..." three turns later, with a made-up product).
    #
    # A prompt saying "only answer the latest" did NOT fix it. Restructuring
    # did: history goes in HERE, plainly marked as already handled, and only
    # the current question is sent as a live message. Same information, same
    # token cost -- but now the model can tell what it's being asked.
    #
    # THE LESSON: when a model misbehaves, your first instinct is to add a
    # sentence to the prompt. Usually the better fix is to change the SHAPE of
    # what you send it.
    if history:
        lines = ["\n## CONVERSATION SO FAR", "Already answered. Context only -- never re-answer any of it."]
        for message in history:
            speaker = "Teacher" if message["role"] == "user" else "You"
            lines.append(f"{speaker}: {message['content']}")
        parts.append("\n".join(lines))

    # --- tools, described by the MCP server itself, not hardcoded here ---
    # The FULL docstring goes in, not just the first line. Those docstrings are
    # where the routing rules live ("use class_stats for class questions, not
    # calculate") -- hide them and a small model reliably picks the wrong tool.
    # This costs a few hundred tokens. Buying good decisions with tokens is the
    # single most common trade in agent design.
    if tools:
        lines = ["\n## TOOLS YOU CAN CALL"]
        for tool in tools:
            params = ", ".join(tool.input_schema.get("properties", {}).keys()) or "none"
            description = "\n".join(f"  {l.strip()}" for l in tool.description.strip().splitlines())
            lines.append(f"- {tool.name}({params}):\n{description}")
        lines.append("- load_skill(name): load detailed instructions for a task.")
        if subagents:
            lines.append(
                "- delegate(agent, task): hand an ENTIRE job to a specialist "
                "subagent and get finished work back."
            )
        parts.append("\n".join(lines))

    # --- subagents: specialists the agent can hand a whole job to ------------
    # Same progressive-disclosure trick as skills: the parent sees a one-line
    # description, never the specialist's persona, procedure, or tool results.
    # That is the entire saving -- the parent is told WHO to ask, not HOW to do it.
    if subagents:
        lines = ["\n## SPECIALISTS YOU CAN DELEGATE TO"]
        lines += [f"- {s['name']}: {s['description']}" for s in subagents]
        lines.append(
            "If the teacher's CURRENT message asks for one of these, call "
            "delegate(agent, task) instead of doing it yourself -- the "
            "specialist knows the procedure and you do not. Put the specialist's "
            "name in `agent`, and in `task` write one sentence naming the "
            "student and why you are reaching out. Do NOT look the student up "
            "first; the specialist does that itself."
        )
        parts.append("\n".join(lines))

    # --- skills: names + descriptions ONLY. Bodies are loaded on demand. ---
    if skills:
        lines = ["\n## SKILLS AVAILABLE"]
        lines += [f"- {s['name']}: {s['description']}" for s in skills]
        lines.append(
            "If the teacher's CURRENT message explicitly asks for one of the "
            "things above (e.g. 'draft an email', 'write me a report'), you "
            "MUST call load_skill(name) BEFORE answering -- you do not know "
            "the correct procedure until you have read it. If they are asking "
            "a question or just sharing information, ignore these entirely; "
            "never start drafting something nobody asked for."
        )
        parts.append("\n".join(lines))

    # --- long-term memory, retrieved for THIS question ---
    if memories:
        lines = ["\n## WHAT YOU REMEMBER (about this teacher and their students)"]
        lines += [f"- {m['text']}" for m in memories]
        lines.append(
            "These are your own private notes, written in the third person. "
            "NEVER paste one back verbatim -- 'The user has a meeting' must "
            "come out as 'You have a meeting'. Rewrite them as natural speech "
            "addressed to the teacher.\n"
            "If any of these is relevant, visibly tailor your answer to it "
            "(e.g. weigh a student's circumstances when judging their numbers). "
            "A fact about a named student applies to THAT student only -- never "
            "carry it over to anyone else. Never announce that you are 'using "
            "memory' -- just sound like you already knew."
        )
        parts.append("\n".join(lines))

    # --- finished work handed back by a subagent -----------------------------
    # Compare the SIZE of this block to what the subagent had in context to
    # produce it (the Subagent panel shows both numbers). That ratio is the
    # entire argument for delegation.
    for agent_name, work in delegated:
        parts.append(
            f"\n## FINISHED WORK FROM YOUR SPECIALIST ({agent_name})\n{work}\n\n"
            "This is finished work. The specialist already checked it against "
            "the gradebook. Reproduce it EXACTLY as written, word for word, as "
            "your reply. Start at its FIRST line -- if it opens with a subject "
            "line, keep the subject line. Do not rewrite it, shorten it, "
            "reorder it, summarise it, or add any fact to it. You may put at "
            "most ONE short sentence before it, and nothing after. Never "
            "mention the specialist, "
            "delegation, or that anyone else was involved.\n"
            "This OVERRIDES the 2-4 sentence limit above -- reproduce the work "
            "in full, however long it is."
        )

    # --- a skill body, once loaded. Note how much bigger this is. ---
    for skill_name, body in loaded_skills:
        parts.append(f"\n## SKILL INSTRUCTIONS: {skill_name}\nFollow these exactly.\n\n{body}")

    # --- results of tools we already called this turn ---
    if observations:
        lines = ["\n## TOOL RESULTS (this turn)"]
        lines += [f"### {name}\n{result}" for name, result in observations]
        # NOTE: an earlier version added "if you don't have the number, say you
        # need to look it up." It backfired badly -- the agent started replying
        # "I'll call query_grades" instead of answering, because it contradicts
        # ANSWER_INSTRUCTION. Two rules that each sound sensible can fight each
        # other. Change one line here and re-run the demo script before keeping it.
        lines.append(
            "These results are authoritative FOR WHAT THEY COVER. Quote the "
            "actual VALUES -- names, numbers, dates -- and never describe what "
            "a tool does or what it 'shows'. If the results list several "
            "students, include every one.\n"
            "CRITICAL: if these results do not actually answer what the teacher "
            "asked, say nothing about them at all. If the teacher asked no "
            "question at all, they asked for no data: acknowledge what they "
            "said and stop, no matter what is listed here. A tool call that "
            "missed is "
            "not evidence -- reciting an unrelated due date or class average "
            "reads as a non-sequitur. Never claim these results explain "
            "something they do not mention (they say nothing about the "
            "teacher's schedule, meetings, or reasons). In that case answer "
            "from what you remember and from the conversation instead."
        )
        parts.append("\n".join(lines))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# STEP A: DECIDE — "call a tool, or answer?"
# ---------------------------------------------------------------------------
def build_decision_schema(tools, tool_names: list[str], skills: list[dict], subagents=()) -> dict:
    """Build the JSON Schema that constrains the DECIDE call.

    Note this is generated from the MCP server's tool list at runtime -- add a
    tool to the server and this schema grows to match, automatically.
    """
    # "name" belongs to load_skill. Restricting it to an enum of real skill
    # names means the model cannot invent (or omit the value of) a skill.
    arg_properties = {
        "name": {"type": "string", "enum": [s["name"] for s in skills] or [""]},
        # `agent` and `task` belong to delegate. Same enum trick again: the model
        # cannot delegate to a specialist that does not exist.
        "agent": {"type": "string", "enum": [s["name"] for s in subagents] or [""]},
        "task": {"type": "string"},
    }
    for tool in tools:
        for arg_name, spec in tool.input_schema.get("properties", {}).items():
            arg_properties[arg_name] = {"type": spec.get("type", "string")}

    # ---- MODIFY HERE ----
    # Two constraints doing real work here:
    #
    #   `enum` on `tool`  -> the model CANNOT emit a tool name that isn't in
    #                        the list. Hallucinated tool names stop being a bug
    #                        you catch and become structurally impossible.
    #
    #   `additionalProperties: False` on `args` -> without this, `args` is an
    #                        open-ended object and the grammar happily lets the
    #                        model generate huge nested JSON until it runs out
    #                        of tokens. (Ask us how we know.) Pinning the allowed
    #                        keys keeps the decision small and fast.
    return {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "tool": {"type": "string", "enum": tool_names + ["none"]},
            "args": {
                "type": "object",
                "properties": arg_properties,
                "additionalProperties": False,
            },
        },
        "required": ["reasoning", "tool", "args"],
    }


def decide(system_prompt: str, user_text: str, schema: dict) -> dict:
    """One constrained-JSON call. Returns {"reasoning":..., "tool":..., "args":{...}}.

    Note what is NOT passed in: the message history. It's already in the system
    prompt, labelled as answered. The only live message is the current question,
    so "what should I do next?" has exactly one possible subject.
    """
    instruction = (
        f'{user_text}\n\n'
        "---\n"
        "Decide the NEXT action for the message above, and nothing else.\n"
        "Pick a tool ONLY if that message needs gradebook information you do "
        'not already have. Otherwise pick "none".\n'
        'Pick "none" when: the WHAT YOU REMEMBER section already answers it; '
        "the CONVERSATION SO FAR already answers it (short follow-ups like "
        '"why?" are almost always this); the TOOL RESULTS section already has '
        "what you need; or I am just chatting or sharing information.\n"
        "Never reach for a tool just to have something to do. A tool that "
        "cannot answer the question is worse than no tool at all -- the "
        "gradebook knows nothing about my schedule, my meetings, or my "
        "personal life.\n"
        'Note: if I say "calculate" about a class average or statistics, the '
        "right tool is class_stats -- the calculate tool is only for raw "
        "arithmetic on numbers already in front of you.\n"
        "Put the tool's arguments in `args` (use {} if it takes none).\n"
        "Keep `reasoning` to one short sentence."
    )

    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        format=schema,
        options={
            "temperature": config.TEMPERATURE,
            # Hard ceiling on the decision. It should be ~40 tokens; if the
            # model ever runs away, we cut it off instead of hanging the demo.
            "num_predict": config.MAX_DECISION_TOKENS,
        },
        keep_alive=config.KEEP_ALIVE,
    )

    try:
        decision = json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        # Should be impossible with constrained decoding, but never trust a
        # model's output without a guard around it.
        return {"reasoning": "unparseable", "tool": "none", "args": {}}

    if not isinstance(decision.get("args"), dict):
        decision["args"] = {}
    return decision


class _SkillTool:
    """Makes `load_skill` look like an MCP tool so it can reuse repair_args().

    MCP tools carry their own schema; load_skill is ours, so we write its schema
    by hand -- an enum of the skills that actually exist on disk.
    """

    def __init__(self, skill_names: list[str]):
        self.name = "load_skill"
        self.input_schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": skill_names or [""]}},
            "required": ["name"],
        }


class _DelegateTool:
    """Same idea as _SkillTool: makes `delegate` look like an MCP tool so it can
    reuse repair_args(). Both `agent` and `task` are required -- a delegation
    with no briefing is the most common way this pattern fails."""

    def __init__(self, agent_names: list[str]):
        self.name = "delegate"
        self.input_schema = {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": agent_names or [""]},
                "task": {"type": "string"},
            },
            "required": ["agent", "task"],
        }


def repair_args(system_prompt: str, user_text: str, tool) -> dict:
    """Ask again for JUST this tool's arguments, using the tool's OWN schema.

    Why this exists: the decide() schema has to cover every tool at once, so it
    can't mark any single argument as required -- and a small model will
    sometimes pick `calculate` and then forget to include `expression`.

    The fix is nice: `tool.input_schema` came from the MCP server, generated
    from the type hints on the function itself. It knows exactly what's
    required. Hand that to the model as a grammar and the argument cannot be
    missing.
    """
    response = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{user_text}\n\n---\nGive the arguments for calling "
                f"`{tool.name}` to answer the message above.",
            },
        ],
        format=tool.input_schema,
        options={"temperature": config.TEMPERATURE, "num_predict": config.MAX_DECISION_TOKENS},
        keep_alive=config.KEEP_ALIVE,
    )
    try:
        return json.loads(response["message"]["content"])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# STEP B + THE LOOP
# ---------------------------------------------------------------------------
def run_turn(mcp, messages: list[dict], user_text: str, retrieval_mode: str, use_subagent: bool = True):
    """Run one full turn. A GENERATOR, so the GUI can render progress live.

    Yields (kind, payload):
        ("trace", str)   -> a line for the Trace panel
        ("chart", str)   -> path to a PNG a tool just rendered; GUI shows it
        ("subagent", d)  -> live state of a subagent; GUI shows it in its own tab
        ("token", str)   -> a piece of the streaming answer
        ("stats", dict)  -> prompt token count
        ("done", list)   -> the updated message list
    """

    # === 1. RECALL ========================================================
    # Before thinking, remember. Pull relevant durable facts out of long-term
    # memory and into the prompt.
    memories = memory.retrieve(user_text, retrieval_mode)
    if memories:
        yield ("trace", f"**Recall** ({retrieval_mode}): found {len(memories)} memory(s)")
        for m in memories:
            yield ("trace", f"&nbsp;&nbsp;&nbsp;&nbsp;#{m['id']} {m['text']}")
    else:
        yield ("trace", f"**Recall** ({retrieval_mode}): nothing relevant found")

    # === 2. SHORT-TERM MEMORY =============================================
    messages.append({"role": "user", "content": user_text})
    before = len(messages)
    messages = trim_short_term(messages)
    if len(messages) < before:
        yield ("trace", f"**Context full** - dropped {before - len(messages)} old message(s). Those facts are GONE.")

    # Everything EXCEPT the current question. This goes into the system prompt
    # as labelled history; the current question is sent on its own. See the
    # CONVERSATION SO FAR block in build_system_prompt() for why.
    history = messages[:-1]

    # Skills are only offered when this message actually asks for something to
    # be produced -- see skills_on_offer(). When none are on offer, load_skill
    # is removed from the tool enum entirely, so it is not merely discouraged;
    # it is ungeneratable.
    offered = skills_on_offer(user_text, skills_loader.list_skills())

    # ---- THE A/B SWITCH (the "Delegate to subagent" toggle in the GUI) ----
    # ON  -> the specialist owns check-in-email. The parent is not even shown
    #        that skill; its only route to the job is delegate(email-writer,...).
    #        One capability, one home -- see subagent.owned_skills().
    # OFF -> the skill goes back to the parent and it does the whole job itself,
    #        exactly as it did before this week.
    #
    # Ask for the same email twice, once each way, and watch the "Prompt size"
    # counter under the chat. Same answer, different amount of the main agent's
    # context spent on it. That difference is the entire lesson, and this toggle
    # is how you show it instead of asserting it.
    if use_subagent:
        owned = subagent.owned_skills()
        skills = [s for s in offered if s["name"].lower() not in owned]
        # Subagents ride the same gate as skills: only on the table when the
        # teacher is actually asking for something to be produced. The rest of
        # the time `delegate` is not merely discouraged -- it is absent from the
        # enum, so it is ungeneratable. Same lesson as everywhere else in this
        # file: if a rule matters, enforce it in code.
        subagents = subagent.list_subagents() if offered else []
    else:
        skills, subagents = offered, []

    tool_names = (
        [t.name for t in mcp.tools]
        + (["load_skill"] if skills else [])
        + (["delegate"] if subagents else [])
    )
    schema = build_decision_schema(mcp.tools, tool_names, skills, subagents)
    skill_names = {s["name"].lower() for s in skills}
    if skills:
        yield ("trace", f"**Skills on offer**: {', '.join(s['name'] for s in skills)}")
    if subagents:
        yield ("trace", f"**Specialists on offer**: {', '.join(s['name'] for s in subagents)}")

    observations: list[tuple[str, str]] = []
    loaded_skills: list[tuple[str, str]] = []
    delegated: list[tuple[str, str]] = []

    # Small models LOVE to call the same tool over and over. Telling them not to
    # in the prompt does not work. So we enforce it in Python instead.
    # General lesson: if a rule actually matters, put it in code, not in a prompt.
    already_called: set[str] = set()
    charts_shown: set[str] = set()

    # === 3. THE LOOP ======================================================
    for step in range(config.MAX_TOOL_STEPS):
        system_prompt = build_system_prompt(
            mcp.tools, memories, skills, loaded_skills, observations, history,
            subagents, delegated,
        )
        decision = decide(system_prompt, user_text, schema)

        tool = decision["tool"]
        yield ("trace", f"**Step {step + 1}** - thinking: _{decision['reasoning'][:150]}_")

        if tool == "none":
            yield ("trace", f"**Step {step + 1}** - no tool needed, answering")
            break

        # -- did the model forget a required argument? ---------------------
        tool_obj = next((t for t in mcp.tools if t.name == tool), None)
        if tool_obj is not None:
            required = tool_obj.input_schema.get("required", [])
            if any(arg not in decision["args"] for arg in required):
                yield ("trace", f"&nbsp;&nbsp;&nbsp;&nbsp;missing required arg — re-asking using `{tool}`'s own schema")
                decision["args"] = repair_args(system_prompt, user_text, tool_obj)

        # Same repair for load_skill, whose "schema" is just the list of skills.
        if tool == "load_skill" and decision["args"].get("name", "").lower() not in skill_names:
            decision["args"] = repair_args(
                system_prompt,
                user_text,
                _SkillTool([s["name"] for s in skills]),
            )

        # ...and for delegate. A missing `task` is the classic delegation bug:
        # the parent picks the right specialist and then tells it nothing.
        if tool == "delegate" and (
            subagent.find(decision["args"].get("agent", "")) is None
            or not str(decision["args"].get("task", "")).strip()
        ):
            yield ("trace", "&nbsp;&nbsp;&nbsp;&nbsp;incomplete briefing — re-asking using `delegate`'s own schema")
            decision["args"] = repair_args(
                system_prompt,
                user_text,
                _DelegateTool([s["name"] for s in subagents]),
            )

        signature = tool + json.dumps(decision["args"], sort_keys=True)
        if signature in already_called:
            yield ("trace", f"**Step {step + 1}** - `{tool}` already called with these args, answering now")
            break
        already_called.add(signature)

        # -- load_skill is handled by US, not the MCP server. --------------
        # Skills are a CLIENT-side capability (instructions we own); MCP tools
        # are SERVER-side capabilities (things someone else provides). A real
        # agent merges tools from several sources like this all the time.
        if tool == "load_skill":
            name = str(decision["args"].get("name", ""))
            # Guard: if even the repair call above failed to produce a real
            # skill name, skip rather than polluting the prompt with an error.
            if name.lower() not in skill_names:
                yield ("trace", f"**Step {step + 1}** - asked for unknown skill `{name}`, ignoring")
                continue
            body = skills_loader.load_skill(name)
            loaded_skills.append((name, body))
            yield ("trace", f"**Step {step + 1}** - loaded skill `{name}` (+~{len(body) // 4} tokens)")
            continue

        # -- delegate is ALSO handled by US, and this is the interesting one --
        # load_skill pulls text INTO this context. delegate does the opposite:
        # it starts a whole second agent loop somewhere else, lets that loop
        # spend its own context, and brings back only the result. Watch the
        # Subagent tab while this runs.
        if tool == "delegate":
            spec = subagent.find(decision["args"].get("agent", ""))
            task = str(decision["args"].get("task", "")).strip()
            if spec is None or not task:
                yield ("trace", f"**Step {step + 1}** - delegation had no valid target, skipping")
                continue

            # THE BRIEFING. A subagent starts with an empty context -- it has
            # never seen this conversation and cannot read our memory. Anything
            # it needs, we hand it. Forget this line and the email is factually
            # fine and humanly tone-deaf: it will not know about the concussion.
            briefing = [m["text"] for m in memories]

            yield ("trace", f"**Step {step + 1}** - `delegate(agent={spec['name']!r}, task={task!r})`")
            if briefing:
                yield ("trace", f"&nbsp;&nbsp;│&nbsp;&nbsp;briefing it with {len(briefing)} memory(s) — it cannot see ours")

            work = ""
            for kind, payload in subagent.run(mcp, spec, task, briefing):
                if kind == "trace":
                    # Indented so the nesting is obvious at a glance: these lines
                    # happened inside a different agent, not this one.
                    yield ("trace", f"&nbsp;&nbsp;│&nbsp;&nbsp;{payload}")
                elif kind == "panel":
                    yield ("subagent", payload)
                elif kind == "result":
                    work = payload

            delegated.append((spec["name"], work))
            yield ("trace", f"**Step {step + 1}** - specialist done; handing its work to the teacher")
            # The specialist finished the job. There is nothing left to gather,
            # so stop looping and go write the reply.
            break

        # -- everything else goes over MCP to the server subprocess ---------
        yield ("trace", f"**Step {step + 1}** - calling `{tool}({json.dumps(decision['args'])})` via MCP")
        try:
            result = mcp.call_tool(tool, decision["args"])
        except Exception as e:
            result = f"Tool failed: {e}"

        # A tool can return an ARTIFACT, not just text. chart_grades renders a
        # PNG and tells us where it is on its first line; we hand the path to
        # the GUI and give the model only the text half. The model never sees
        # the image -- it doesn't need to. It saw the numbers.
        if result.startswith("CHART_SAVED: "):
            first_line, _, rest = result.partition("\n")
            path = first_line.removeprefix("CHART_SAVED: ").strip()
            # Only show each chart once. `chart_grades({})` and
            # `chart_grades({"target": "class"})` are different signatures to
            # the repeat-guard above but render the same PNG, and the teacher
            # does not want the same image twice in the chat.
            if path not in charts_shown:
                charts_shown.add(path)
                yield ("chart", path)
            result = "(Chart rendered -- it is already displayed to the teacher.)\n" + rest

        observations.append((tool, result))
        yield ("trace", f"&nbsp;&nbsp;&nbsp;&nbsp;-> {result[:200]}")

    # === 4. ANSWER (streaming) ===========================================
    # Same shape as decide(): history lives in the system prompt, and the only
    # live message is what the teacher actually just asked.
    system_prompt = build_system_prompt(
        mcp.tools, memories, skills, loaded_skills, observations, history,
        subagents, delegated,
    )
    stream = ollama.chat(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": system_prompt + ANSWER_INSTRUCTION},
            {"role": "user", "content": user_text},
        ],
        stream=True,
        keep_alive=config.KEEP_ALIVE,
    )

    answer, prompt_tokens = "", 0
    for chunk in stream:
        piece = chunk["message"]["content"]
        answer += piece
        yield ("token", piece)
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count", 0)

    messages.append({"role": "assistant", "content": answer})
    messages = trim_short_term(messages)
    yield (
        "stats",
        {
            "prompt_tokens": prompt_tokens,
            "skills_loaded": len(loaded_skills),
            "delegations": len(delegated),
        },
    )

    # === 5. REFLECT — write to long-term memory ==========================
    # This happens AFTER the student already has their answer, so the extra
    # LLM call never makes the reply feel slow.
    yield ("trace", "**Reflect** - checking for durable facts...")
    facts = memory.extract_facts(user_text, answer)
    if not facts:
        yield ("trace", "&nbsp;&nbsp;&nbsp;&nbsp;nothing worth remembering (this is normal)")
    for fact in facts:
        # remember() does the add/update/skip reconciliation -- see memory.py
        yield ("trace", f"&nbsp;&nbsp;&nbsp;&nbsp;{memory.remember(fact)}")

    yield ("done", messages)
