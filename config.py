"""
config.py — Every knob in the system, in one place.

If you want to change how the demo behaves, change it HERE first.
Students: this is the file to experiment with before touching anything else.
"""

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
# The "brain". Must be pulled first:  ollama pull gemma3:4b
#
# Machine struggling? Swap to a smaller model from https://ollama.com/library
#   ollama pull gemma3:1b     -> MODEL = "gemma3:1b"     (fast, dumber)
#   ollama pull qwen3:4b      -> MODEL = "qwen3:4b"      (better at tools)
# Nothing else in the codebase needs to change.
MODEL = "qwen3:4b"
# MODEL = "nemotron-3.5-lightning"

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


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY (facts on disk)
# ---------------------------------------------------------------------------
MEMORY_FILE = "memories.json"

# How many remembered facts get injected into the prompt each turn.
RETRIEVAL_TOP_K = 3

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
# SUBAGENTS (see subagent.py)
# ---------------------------------------------------------------------------
# A subagent gets its own, SHORTER leash than the parent. It has one job and one
# or two tools; if it hasn't gathered what it needs in two steps, it is lost, and
# more steps will not save it -- they will just make the demo slow. Every nested
# agent multiplies your worst-case latency, so cap the child tighter than the
# parent and mean it.
SUBAGENT_MAX_TOOL_STEPS = 2

# The MCP server we launch as a subprocess. Add more servers here later.
MCP_SERVER_SCRIPT = "course_server.py"

# Where SKILL.md files live.
SKILLS_DIR = "skills"
