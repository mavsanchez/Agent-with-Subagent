"""
setup_check.py — Run this BEFORE class.

    uv run python setup_check.py

Checks every moving part and tells you the exact command to fix anything
that's broken. If all the lines are green, the demo will run.
"""

import importlib.util
import sys

from teacher_assistant import settings

OK, BAD = "  [ OK ]", "  [FAIL]"
problems = []


def check(label, fn, fix):
    """Run one check. Print a green/red line. Remember the fix if it failed."""
    try:
        detail = fn()
        print(f"{OK} {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        print(f"{BAD} {label} — {e}")
        problems.append(fix)
        return False


print(f"\nPython {sys.version.split()[0]}  ({sys.executable})\n")
if sys.version_info < (3, 10):
    problems.append("Python 3.10+ required:  uv venv --python 3.11 && uv sync")
    print(f"{BAD} Python version — need 3.10 or newer")


# --- 1. packages -----------------------------------------------------------
def _packages():
    missing = [
        name
        for name in (
            "openai", "mcp", "gradio", "matplotlib", "requests", "bs4",
        )
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "openai, mcp, gradio, matplotlib, requests, beautifulsoup4"


have_packages = check("Python packages", _packages, "uv sync")


# --- 2. is LiteLLM configured and reachable? ------------------------------
def _llm_environment():
    if not settings.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set")
    return f"{settings.LLM_BASE_URL}, model alias '{settings.LLM_MODEL}'"


configured = check(
    "LLM environment",
    _llm_environment,
    "Set LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY (see .env.example)",
)


def _litellm_server():
    from teacher_assistant import llm

    llm.check_connection()
    return f"reachable at {settings.LLM_BASE_URL}"


server_up = False
if have_packages and configured:
    server_up = check(
        "LiteLLM chat route",
        _litellm_server,
        f"Check LiteLLM, DNS/network access, and credentials for {settings.LLM_BASE_URL}",
    )


# --- 3. does MCP start and does the client expose all capabilities? --------
def _mcp_server():
    from teacher_assistant.mcp.client import MCPClient

    client = MCPClient(settings.MCP_SERVER_PATH)
    try:
        client.connect()
        names = [t.name for t in client.tools]
    finally:
        client.close()
    if not names:
        raise RuntimeError("servers started but advertised no tools")
    if settings.WEB_RESEARCH_TOOL not in names:
        raise RuntimeError("client did not expose web_research")
    return ", ".join(names)


if server_up:
    check(
        "MCP and web-research tools",
        _mcp_server,
        "Check teacher_assistant/mcp/client.py and course_server.py for errors",
    )


# --- summary ---------------------------------------------------------------
print()
if problems:
    print("Not ready yet. Run these:\n")
    for p in problems:
        print(f"    {p}")
    print()
    sys.exit(1)

print("All checks passed. Start the demo with:\n\n    uv run python app.py\n")
