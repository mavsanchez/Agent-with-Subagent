"""
setup_check.py — Run this BEFORE class.

    python setup_check.py

Checks every moving part and tells you the exact command to fix anything
that's broken. If all the lines are green, the demo will run.
"""

import importlib.util
import sys

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
    problems.append("Python 3.10+ required:  conda create -n agentdemo python=3.11")
    print(f"{BAD} Python version — need 3.10 or newer")


# --- 1. packages -----------------------------------------------------------
def _packages():
    missing = [
        name
        for name in ("ollama", "mcp", "gradio", "numpy", "matplotlib")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "ollama, mcp, gradio, numpy, matplotlib"


have_packages = check("Python packages", _packages, "pip install -r requirements.txt")


# --- 2. is the Ollama server running? --------------------------------------
# We check by talking to it, not by looking for the binary on PATH -- on
# Windows the app is often running fine while `ollama` isn't on PATH.
installed_models = []


def _ollama_server():
    global installed_models
    import ollama

    installed_models = [m.model for m in ollama.list().models]
    return f"reachable, {len(installed_models)} model(s) installed"


server_up = False
if have_packages:
    server_up = check(
        "Ollama server",
        _ollama_server,
        "Start Ollama (install from https://ollama.com/download), then: ollama serve",
    )


# --- 3. are the models pulled? ---------------------------------------------
def _model_present(name):
    def inner():
        # `ollama list` reports "gemma3:4b"; a user may have configured "gemma3".
        if not any(m == name or m.split(":")[0] == name.split(":")[0] for m in installed_models):
            raise RuntimeError("not pulled")
        return None

    return inner


if server_up:
    import config

    check(f"Model '{config.MODEL}'", _model_present(config.MODEL), f"ollama pull {config.MODEL}")
    check(
        f"Embeddings '{config.EMBED_MODEL}'",
        _model_present(config.EMBED_MODEL),
        f"ollama pull {config.EMBED_MODEL}",
    )


# --- 4. does the MCP server actually start and expose tools? ---------------
def _mcp_server():
    from mcp_client import MCPClient

    import config

    client = MCPClient(config.MCP_SERVER_SCRIPT)
    client.connect()
    names = [t.name for t in client.tools]
    client.close()
    if not names:
        raise RuntimeError("server started but advertised no tools")
    return ", ".join(names)


if server_up:
    check("MCP server", _mcp_server, "Check course_server.py for syntax errors")


# --- summary ---------------------------------------------------------------
print()
if problems:
    print("Not ready yet. Run these:\n")
    for p in problems:
        print(f"    {p}")
    print()
    sys.exit(1)

print("All checks passed. Start the demo with:\n\n    python app.py\n")
