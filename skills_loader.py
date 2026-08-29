"""
skills_loader.py — SKILLS, and progressive disclosure.

MEMORY vs SKILLS, the distinction worth putting on a slide:

    Long-term memory = facts.      "The user is a senior."        (knowing THAT)
    Skills           = procedures. "Here's how to build a plan."  (knowing HOW)

Psychologists call these semantic and procedural memory. Your agent needs both.

PROGRESSIVE DISCLOSURE
----------------------
A skill can be long. If you paste every skill into the system prompt, you burn
your whole context window before the user says hello -- and on a 4B model with
a small window, that isn't a rounding error, it's the difference between
working and not.

So we do what Claude Code does:
  - At startup, load ONLY each skill's name + description (~15 tokens each).
  - The agent gets a `load_skill` tool.
  - The full body is pulled in ONLY when the agent decides it's needed.

Watch the token counter in the GUI when you trigger one. That gap is the lesson.

FILE FORMAT (skills/<name>/SKILL.md):
    ---
    name: study-plan
    description: When to use this skill. The agent reads THIS to decide.
    ---
    Markdown instructions the agent follows once loaded.
"""

from pathlib import Path

import config

SKILLS_PATH = Path(__file__).parent / config.SKILLS_DIR


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into its YAML-ish header and its body.

    We hand-parse the two fields we need instead of pulling in PyYAML. One
    fewer dependency, and you can read the whole parser in ten seconds.
    """
    if not text.startswith("---"):
        return {}, text

    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta, body.strip()


def list_skills() -> list[dict]:
    """Return name + description for every skill. This is the CHEAP part."""
    skills = []
    if not SKILLS_PATH.exists():
        return skills

    for skill_file in sorted(SKILLS_PATH.glob("*/SKILL.md")):
        meta, _ = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        skills.append(
            {
                "name": meta.get("name", skill_file.parent.name),
                "description": meta.get("description", ""),
                "path": skill_file,
            }
        )
    return skills


def load_skill(name: str) -> str:
    """Return a skill's full body. This is the EXPENSIVE part, loaded on demand."""
    for skill in list_skills():
        if skill["name"].lower() == name.strip().lower():
            _, body = _parse_frontmatter(skill["path"].read_text(encoding="utf-8"))
            return body
    available = ", ".join(s["name"] for s in list_skills())
    return f"No skill named '{name}'. Available: {available}"
