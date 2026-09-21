"""
settings.py — Every knob and project path in the system, in one place.

If you want to change how the demo behaves, change it HERE first.
Students: this is the file to experiment with before touching anything else.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------
# Resolve paths from this package instead of the process working directory.
# That keeps both entry points reliable when they are launched from elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
MCP_SERVER_PATH = (
    PROJECT_ROOT / "teacher_assistant" / "mcp" / "course_server.py"
).resolve()
MEMORY_PATH = (
    PROJECT_ROOT / "teacher_assistant" / "memory" / "memories.json"
).resolve()
SKILLS_PATH = (PROJECT_ROOT / "teacher_assistant" / "skills").resolve()

# ---------------------------------------------------------------------------
# KEY-FREE WEB RESEARCH
# ---------------------------------------------------------------------------
# MCPClient exposes this local capability beside the tools discovered from the
# course MCP server. It searches public result pages and requires no API key.
WEB_RESEARCH_TOOL = "web_research"
WEB_SEARCH_RESULT_COUNT = 5
WEB_RESEARCH_TIMEOUT_SECONDS = 15

# ---------------------------------------------------------------------------
# LITELLM / DGX SPARK
# ---------------------------------------------------------------------------
# LiteLLM resolves this alias and routes requests to vLLM on the DGX Spark.
# Override either value through the environment without changing application code.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://dgx-ramona:4000/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "agent")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_TIMEOUT_SECONDS = 120.0

# Includes hidden reasoning tokens emitted by reasoning models before the
# teacher-facing answer. A larger cap prevents a blank answer on complex turns.
MAX_ANSWER_TOKENS = 8192
LLM_REASONING_EFFORT = "low"

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
SHORT_TERM_MAX_TURNS = 500


# How many remembered facts get injected into the prompt each turn.
RETRIEVAL_TOP_K = 3

# ---------------------------------------------------------------------------
# THE AGENT LOOP
# ---------------------------------------------------------------------------
# Max tool calls before we force the agent to answer. A backstop -- the loop
# also stops early if the model tries to repeat a call it already made.
MAX_TOOL_STEPS = 3

# Hard token cap on the "which tool?" decision. That JSON should be ~40 tokens.
# Without a ceiling, a small model that starts rambling can stall the whole demo.
# The DGX reasoning model uses part of this budget for hidden reasoning before
# emitting its small JSON decision, so this needs more headroom than the JSON.
MAX_DECISION_TOKENS = 4096

# ---------------------------------------------------------------------------
# SUBAGENTS (see agents/subagent.py)
# ---------------------------------------------------------------------------
# The current specialist has one job and needs exactly one student_report call.
# A single step prevents an unnecessary second routing pass while keeping the
# child more tightly bounded than the parent.
SUBAGENT_MAX_TOOL_STEPS = 1
