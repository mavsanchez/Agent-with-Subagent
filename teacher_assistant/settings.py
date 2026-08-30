"""
settings.py — Every knob and project path in the system, in one place.

If you want to change how the demo behaves, change it HERE first.
Students: this is the file to experiment with before touching anything else.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
# Resolve paths from this package instead of the process working directory.
# That keeps both entry points reliable when they are launched from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_PATH = (
    PROJECT_ROOT / "teacher_assistant" / "mcp" / "course_server.py"
).resolve()
MEMORY_PATH = (
    PROJECT_ROOT / "teacher_assistant" / "memory" / "memories.json"
).resolve()
SKILLS_PATH = (PROJECT_ROOT / "teacher_assistant" / "skills").resolve()

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
# The "brain". Must be pulled first:  ollama pull gemma3:4b
#
# Machine struggling? Swap to a smaller model from https://ollama.com/library
#   ollama pull gemma3:1b     -> MODEL = "gemma3:1b"     (fast, dumber)
#   ollama pull qwen3:4b      -> MODEL = "qwen3:4b"      (better at tools)
# Nothing else in the codebase needs to change.
MODEL = "gemma3:4b"
# MODEL = "nemotron-3.5-lightning"

# Keep every chat call inside a modest shared context window so Ollama does not
# allocate the model's much larger default context. Final prose is capped
# separately from the short, structured routing decisions below.
MODEL_CONTEXT_TOKENS = 8192
MAX_ANSWER_TOKENS = 384


def chat_options(**overrides) -> dict:
    """Return fresh Ollama chat options with the shared context configured."""
    return {"num_ctx": MODEL_CONTEXT_TOKENS, **overrides}


# Turns text into vectors so we can search memory by MEANING, not keywords.
#   ollama pull nomic-embed-text
EMBED_MODEL = "bge-m3"
# EMBED_MODEL = "qwen3-embedding:0.6b"
# EMBED_MODEL = "nomic-embed-text"

# Keeps the model loaded in RAM between calls. Without this, Ollama unloads it
# after ~5 min and the next question takes 10+ extra seconds. Demo killer.
KEEP_ALIVE = "10m"

# 0 = deterministic. We want the tool-choosing step to be boring and repeatable,
# especially in front of a live audience.
TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# SHORT-TERM MEMORY (the conversation)
# ---------------------------------------------------------------------------
# How many back-and-forth exchanges the agent remembers in RAM.
#
# This is set deliberately LOW so we can demo the failure mode live: chat past
# 6 turns and the agent literally forgets what you told it, because those
# messages get dropped before we send them to the model.
#
# Real systems set this to hundreds. The failure is the same, just later.
SHORT_TERM_MAX_TURNS = 6


# How many remembered facts get injected into the prompt each turn.
RETRIEVAL_TOP_K = 3

# Minimum semantic similarity required before a stored fact is recalled.
RETRIEVAL_MIN_SCORE = 0.50

# When a NEW fact is this similar to an OLD one (0-1 cosine), we suspect they
# are about the same thing and ask the model: add, update, or skip?
# This is what stops "I'm vegetarian" and "I eat chicken now" from both living
# in memory forever, contradicting each other.
#
# ---- MODIFY HERE ----
# Tuned from real measured runs, and worth showing students how. Against
# "Priya has an extended-time accommodation for exams":
#   0.85  "Priya no longer needs the accommodation"  <- REAL conflict, must update
#   0.71  "Priya is dealing with a family emergency" <- merely related, must NOT
#   0.43  "Marcus has been dealing with a concussion"<- different topic entirely
# At 0.80 the real conflict fires and the related fact doesn't. Set it too high
# and contradictions pile up; too low and memories erase each other.
SIMILARITY_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# THE AGENT LOOP
# ---------------------------------------------------------------------------
# Max tool calls before we force the agent to answer. A backstop -- the loop
# also stops early if the model tries to repeat a call it already made.
MAX_TOOL_STEPS = 3

# Hard token cap on the "which tool?" decision. That JSON should be ~40 tokens.
# Without a ceiling, a small model that starts rambling can stall the whole demo.
MAX_DECISION_TOKENS = 200

# ---------------------------------------------------------------------------
# SUBAGENTS (see agents/subagent.py)
# ---------------------------------------------------------------------------
# The current specialist has one job and needs exactly one student_report call.
# A single step prevents an unnecessary second routing pass while keeping the
# child more tightly bounded than the parent.
SUBAGENT_MAX_TOOL_STEPS = 1
