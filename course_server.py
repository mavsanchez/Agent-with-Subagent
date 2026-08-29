"""
course_server.py — An MCP server.

WHAT IS THIS FILE?
------------------
This is a completely separate PROGRAM. The agent does not import it. Instead,
the agent launches it as a subprocess and talks to it over stdin/stdout using
JSON-RPC. That sounds like overkill for a handful of functions -- and for THIS
demo, honestly, it is. But it buys three things that matter in real systems:

  1. DISCOVERY. The agent never hardcodes tool names. It asks the server
     "what can you do?" at startup (list_tools) and builds its prompt from the
     answer. Add a tool here, restart, and the agent can use it -- with zero
     changes to agent.py.

  2. ISOLATION. This could be on another machine, written in TypeScript, or
     maintained by a team you've never met. The agent doesn't care.

  3. REUSE. This exact file, unchanged, can be plugged into Claude Desktop,
     Claude Code, Cursor, or any other MCP client. Write the tool once,
     use it everywhere. That is why people call MCP "USB-C for tools."

THE DIVISION OF LABOR (the actual lesson of this file)
------------------------------------------------------
Every tool below is deterministic Python: exact averages, exact medians, a
real chart rendered by matplotlib. An LLM cannot do ANY of that reliably --
it can't see the gradebook, and it can't be trusted to average 32 numbers.

But look at what the tools DON'T do: none of them can answer "should I be
worried about Sam?" That answer needs the numbers (tools), the trend (tools),
plus judgement and context the teacher has shared (memory). Gluing those
together into an assessment is the LLM's half of the job.

    Tools  -> facts, math, artifacts.   (things code does perfectly)
    LLM    -> interpretation, judgement, language.  (things code can't do at all)

A NOTE ON "FastMCP"
-------------------
If you Google MCP you will find tutorials using `FastMCP`. In v2 of the
official SDK that class was renamed:

    from mcp.server.fastmcp import FastMCP     # old (mcp 1.x)
    from mcp.server import MCPServer           # new (mcp 2.x) -- what we use

Same idea, same decorator, new name.

HOW IT WORKS
------------
The @server.tool() decorator reads your function's TYPE HINTS and DOCSTRING and
auto-generates the JSON schema that gets sent to the agent. So the docstring
below is not a comment -- it is the prompt the LLM reads to decide whether to
call this tool. Write docstrings like you're writing instructions, because
you are.

RUN IT STANDALONE (to prove it's a real server):
    python course_server.py
    (it will sit there waiting for JSON-RPC on stdin -- Ctrl+C to quit)
"""

import json
import statistics
from pathlib import Path

from mcp.server import MCPServer

# The server's name. Shows up in client UIs.
server = MCPServer("gradebook-tools")

DATA_FILE = Path(__file__).parent / "course_data.json"
CHARTS_DIR = Path(__file__).parent / "charts"


# ---------------------------------------------------------------------------
# DATA LAYER
# ---------------------------------------------------------------------------
# We read a JSON file because it's zero-setup and students can open it in an
# editor to see exactly what the agent sees.
#
# ---- MODIFY HERE ----
# In a real project this is where you'd query Postgres, hit your LMS's REST
# API (Canvas has one!), or call an internal service. NOTE: only this function
# changes. The @server.tool() functions below stay identical, and the agent
# never knows the difference. That boundary is the whole point of MCP.
def _load_data() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def _find_student(data: dict, name: str) -> str | None:
    """Forgiving lookup -- small models are sloppy with capitalization, and
    teachers say 'Sam', not 'Sam Rivera'. Match full name, either name part,
    or prefix."""
    name = name.strip().lower()
    if not name:
        return None
    for full in data["students"]:
        parts = full.lower().split()
        if name == full.lower() or name in parts:
            return full
    for full in data["students"]:
        if full.lower().startswith(name):
            return full
    return None


def _short(assignment: str) -> str:
    """'HW2 - Tool Use & Function Calling' -> 'HW2', for chart labels."""
    return assignment.split(" - ")[0]


# ---------------------------------------------------------------------------
# TOOL 1 — the "LLMs can't do math" demo
# ---------------------------------------------------------------------------
@server.tool()
def calculate(expression: str) -> str:
    """Evaluate a math expression and return the exact answer.

    ALWAYS use this for arithmetic. Never do math yourself -- you will get it
    wrong.

    Only use numbers that appear in the conversation or in an earlier tool
    result. NEVER invent numbers to put in the expression.

    Do not use this for class averages or statistics -- class_stats already
    computed those exactly. This tool is for ad-hoc math like grade projections.

    Always parenthesize fully: "(94 + 61 + 88 + 79) / 4", never
    "94 + 61 + 88 + 79 / 4" -- those give different answers.
    """
    # NOTE: eval() is used here for brevity in a teaching demo. It runs on the
    # student's own machine with a stripped-down namespace. In production you
    # would use a real expression parser (e.g. the `asteval` package) -- never
    # eval() a string that came out of an LLM on a server you care about.
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# TOOL 2 — one student, in depth
# ---------------------------------------------------------------------------
@server.tool()
def student_report(student: str) -> str:
    """Full record for ONE student: every grade, their average, attendance,
    and late submissions.

    Use this whenever the teacher asks about a specific student by name --
    "how is Sam doing", "pull up Marcus", "what did Jordan get on HW3".

    This is one student only. For the roster or every student at once, use
    list_students; for class-wide statistics, use class_stats.
    """
    data = _load_data()
    match = _find_student(data, student)
    if match is None:
        known = ", ".join(data["students"])
        return f"No student named '{student}'. Known students: {known}"

    record = data["students"][match]
    scores = list(record["grades"].values())
    lines = [f"Record for {match} ({data['course']}):"]
    lines += [f"  - {a}: {s}/100" for a, s in record["grades"].items()]
    lines.append(f"  Average: {sum(scores) / len(scores):.1f}/100")
    lines.append(f"  Attendance: {record['attendance_pct']}%")
    lines.append(f"  Late submissions: {record['late_submissions']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 3 — the roster
# ---------------------------------------------------------------------------
# This tool exists because of a real failure: asked "who are all my students?",
# the agent had no tool that answered it -- so the model invented a roster with
# confident, wrong numbers. When the agent fabricates, the fix is usually not a
# better prompt. It's a missing tool.
@server.tool()
def list_students() -> str:
    """List EVERY student in the class with their average, attendance, and
    late submissions -- one line each.

    Use this when the teacher asks who their students are, for the roster,
    or for stats/averages of each individual student at once. For one
    student's full grade breakdown, use student_report instead.
    """
    data = _load_data()
    lines = [f"Roster for {data['course']} ({len(data['students'])} students):"]
    for name, rec in data["students"].items():
        scores = list(rec["grades"].values())
        lines.append(
            f"  - {name}: average {sum(scores) / len(scores):.1f}/100, "
            f"attendance {rec['attendance_pct']}%, "
            f"late submissions {rec['late_submissions']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 4 — the whole class, exactly
# ---------------------------------------------------------------------------
@server.tool()
def class_stats(assignment: str = "overall") -> str:
    """Exact summary statistics for the WHOLE CLASS: mean, median, high, low,
    and who scored the high/low, per assignment.

    Use this for ANY question about the class as a whole: "how did the class
    do", "what was the average on HW2", "what's the spread". Never compute
    class statistics yourself -- this tool sees every student, you do not.

    Pass "overall" (the default) for a table covering every assignment.
    Pass part of an assignment name, e.g. "HW2" or "Midterm", for one.
    """
    data = _load_data()
    wanted = assignment.strip().lower()

    # Models naturally say "overall"/"all"/"everything" rather than passing an
    # empty string, so accept all of them. Meeting the model where it is costs
    # one line and removes a whole class of failure.
    if wanted in ("overall", "all", "everything", "total", "average", "any", ""):
        targets = data["assignments"]
    else:
        targets = [a for a in data["assignments"] if wanted in a.lower()]
        if not targets:
            return f"No assignment matching '{assignment}'. Valid: {', '.join(data['assignments'])}"

    n = len(data["students"])
    lines = [f"Class statistics for {data['course']} ({n} students):"]
    for a in targets:
        by_student = {name: rec["grades"][a] for name, rec in data["students"].items()}
        scores = list(by_student.values())
        top = max(by_student, key=by_student.get)
        bottom = min(by_student, key=by_student.get)
        lines.append(
            f"  - {a}: mean {statistics.mean(scores):.1f}, "
            f"median {statistics.median(scores):.1f}, "
            f"high {by_student[top]} ({top}), low {by_student[bottom]} ({bottom})"
        )
    if len(targets) > 1:
        everything = [s for rec in data["students"].values() for s in rec["grades"].values()]
        lines.append(f"  Overall course mean: {statistics.mean(everything):.1f}/100")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 5 — deadlines
# ---------------------------------------------------------------------------
@server.tool()
def list_deadlines() -> str:
    """List all upcoming assignment deadlines with due dates and grade weights.

    Use this when the teacher asks what is due next, what is coming up, or
    anything involving the course schedule.
    """
    data = _load_data()
    lines = [f"Upcoming deadlines for {data['course']}:"]
    lines += [
        f"  - {d['assignment']} - due {d['due']} (worth {d['weight']} of final grade)"
        for d in data["deadlines"]
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TOOL 6 — a tool that returns an ARTIFACT, not just text
# ---------------------------------------------------------------------------
# This is the one to linger on in class. The LLM cannot draw. It never will.
# But it can DECIDE a chart is needed and delegate to code that draws one.
# The tool returns two things: a marker line the GUI uses to display the image,
# and the plotted numbers as text so the LLM can talk about what's in it.
@server.tool()
def chart_grades(target: str = "class") -> str:
    """Draw a bar chart as a PNG image and display it to the teacher.

    Use this whenever the teacher asks for a chart, graph, plot, or any
    "visual" or "picture" of performance.

    Pass "class" (the default) to chart the class average on each assignment.
    Pass a student's name to chart that student's scores side by side with the
    class average.

    The chart is shown to the teacher automatically. Never describe the image
    file itself -- just summarize what the numbers show.
    """
    import matplotlib

    matplotlib.use("Agg")  # no GUI window -- we render straight to a file
    import matplotlib.pyplot as plt

    data = _load_data()
    assignments = data["assignments"]
    labels = [_short(a) for a in assignments]
    class_avgs = [
        statistics.mean(rec["grades"][a] for rec in data["students"].values())
        for a in assignments
    ]

    CHARTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))

    student = None if target.strip().lower() in ("class", "overall", "all", "") else _find_student(data, target)
    if target.strip().lower() not in ("class", "overall", "all", "") and student is None:
        return f"No student named '{target}'. Known students: {', '.join(data['students'])}"

    if student is None:
        ax.bar(labels, class_avgs, color="#4c72b0")
        ax.set_title(f"Class average by assignment — {data['course']}")
        path = CHARTS_DIR / "class_averages.png"
        plotted = [f"  - {l}: class average {v:.1f}" for l, v in zip(labels, class_avgs)]
    else:
        scores = [data["students"][student]["grades"][a] for a in assignments]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], scores, width=0.4, label=student, color="#4c72b0")
        ax.bar([i + 0.2 for i in x], class_avgs, width=0.4, label="Class avg", color="#c4c4c4")
        ax.set_xticks(list(x), labels)
        ax.legend()
        ax.set_title(f"{student} vs class average — {data['course']}")
        path = CHARTS_DIR / f"{student.split()[0].lower()}_vs_class.png"
        # Precompute the comparisons in code. A small model asked to compare 94
        # to 82.5 will sometimes get it backwards; code never does. If a
        # comparison matters, do the comparing in the tool, not the prompt.
        plotted = [
            f"  - {l}: {student} {s}, class average {v:.1f} "
            f"({'ABOVE' if s >= v else 'BELOW'} average by {abs(s - v):.1f})"
            for l, s, v in zip(labels, scores, class_avgs)
        ]
        above = [l for l, s, v in zip(labels, scores, class_avgs) if s >= v]
        below = [l for l, s, v in zip(labels, scores, class_avgs) if s < v]
        plotted.append(
            f"Correct summary (repeat this faithfully): {student} scored above "
            f"the class average on {', '.join(above) or 'nothing'} and below it "
            f"on {', '.join(below) or 'nothing'}."
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score /100")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

    # The GUI watches for this marker line and renders the PNG in the chat.
    return f"CHART_SAVED: {path}\nThe chart plots:\n" + "\n".join(plotted)


# ---------------------------------------------------------------------------
# TOOL 7 — THE LIVE DEMO TOOL
# ---------------------------------------------------------------------------
# Uncomment this function during class, restart the app, and watch the agent
# start using it immediately. You will not touch agent.py. You will not touch
# app.py. The Tools panel in the GUI will just grow another entry.
#
# THAT is runtime tool discovery, and it is the thing MCP is actually for.
#
# Select everything between the two markers below and hit Ctrl+/ (Cmd+/ on Mac).
# Do NOT include this paragraph in the selection.

# >>>>>>>>>> UNCOMMENT FROM HERE >>>>>>>>>>
# @server.tool()
# def find_at_risk() -> str:
#     """List students who may be at risk, with the evidence: course average
#     below 70, attendance below 80%, or a grade trend that is falling.
#
#     Use this when the teacher asks who is struggling, who is falling behind,
#     who needs a check-in, or who they should be worried about. Never guess
#     at this yourself -- this tool applies exact thresholds to every student.
#     """
#     data = _load_data()
#     flagged = []
#     for name, rec in data["students"].items():
#         scores = list(rec["grades"].values())
#         avg = sum(scores) / len(scores)
#         reasons = []
#         if avg < 70:
#             reasons.append(f"average {avg:.1f}")
#         if rec["attendance_pct"] < 80:
#             reasons.append(f"attendance {rec['attendance_pct']}%")
#         # "Trend" = second half of the term vs first half, 12+ points down.
#         half = len(scores) // 2
#         drop = sum(scores[:half]) / half - sum(scores[half:]) / (len(scores) - half)
#         if drop >= 12:
#             reasons.append(f"scores falling (down {drop:.0f} pts)")
#         if reasons:
#             flagged.append(f"  - {name}: {', '.join(reasons)}")
#
#     if not flagged:
#         return "No students currently meet the at-risk criteria."
#     return "At-risk students (avg < 70, attendance < 80%, or falling scores):\n" + "\n".join(flagged)
# <<<<<<<<<< TO HERE <<<<<<<<<<


if __name__ == "__main__":
    # stdio transport: the client launches this file and pipes JSON-RPC over
    # stdin/stdout. MCP also supports HTTP for remote servers -- see the README.
    server.run()
