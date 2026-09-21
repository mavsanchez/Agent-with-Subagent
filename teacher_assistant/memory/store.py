"""
store.py — LONG-TERM MEMORY.

Short-term memory is just a Python list of messages (see agents/main.py). It lives in
RAM, it has a size limit, and when it overflows, facts are gone forever.

Long-term memory is this file: durable facts, written to memories.json, that
survive across turns AND across restarts. Three jobs:

    1. EXTRACT   -- after each turn, ask the LLM "did we learn anything durable?"
    2. RECONCILE -- if the new fact contradicts an old one, UPDATE, don't append.
    3. RETRIEVE  -- pull the few relevant facts back into the next prompt.

Step 2 is the one tutorials skip and real systems die on. Step 3 is where the
keyword-vs-semantic comparison lives.
"""

import json

from teacher_assistant import llm, settings

MEMORY_PATH = settings.MEMORY_PATH


# ---------------------------------------------------------------------------
# STORAGE — it's just a JSON file. Open it in your editor while the demo runs.
# ---------------------------------------------------------------------------
def load() -> list[dict]:
    if not MEMORY_PATH.exists():
        return []
    records = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    # Older versions persisted model-specific vectors. LiteLLM chat-based
    # semantic scoring needs only the durable fact data.
    return [
        {"id": record["id"], "text": record["text"]}
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("id"), int)
        and isinstance(record.get("text"), str)
    ]


def save(memories: list[dict]) -> None:
    MEMORY_PATH.write_text(json.dumps(memories, indent=2), encoding="utf-8")


def wipe() -> None:
    save([])


# ---------------------------------------------------------------------------
# SEMANTIC SCORING — comparing meaning through the chat model
# ---------------------------------------------------------------------------
# The `agent` alias can judge sentences as related even with no words in common.
# This keeps semantic memory on DGX without requiring `/v1/embeddings`.
SEMANTIC_RETRIEVAL_PROMPT = """Select saved memories that are relevant to
answering the query. Judge meaning and topic, not shared words. Return at most
the requested number of IDs, most useful first. Return an empty list when none
are relevant. Output the JSON decision immediately."""


def _semantic_relevant_ids(query: str, memories: list[dict]) -> list[int]:
    """Use the configured chat model to select semantically relevant facts."""
    ids = [memory["id"] for memory in memories]
    schema = {
        "type": "object",
        "properties": {
            "relevant_ids": {
                "type": "array",
                "items": {"type": "integer", "enum": ids},
                "maxItems": settings.RETRIEVAL_TOP_K,
            }
        },
        "required": ["relevant_ids"],
    }
    candidates = "\n".join(
        f"{memory['id']}: {memory['text']}" for memory in memories
    )
    response = llm.chat(
        messages=[
            {"role": "system", "content": SEMANTIC_RETRIEVAL_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Return at most {settings.RETRIEVAL_TOP_K} IDs.\n"
                    f"QUERY: {query}\n\nMEMORIES:\n{candidates}"
                ),
            },
        ],
        response_schema=schema,
        schema_name="relevant_memories",
        temperature=0,
        max_tokens=settings.MAX_DECISION_TOKENS,
    )
    try:
        selected = json.loads(response.content).get("relevant_ids", [])
    except json.JSONDecodeError:
        return []
    valid_ids = set(ids)
    return [item for item in selected if item in valid_ids][: settings.RETRIEVAL_TOP_K]


# ---------------------------------------------------------------------------
# 1. EXTRACT — "did we learn anything worth keeping?"
# ---------------------------------------------------------------------------
EXTRACT_PROMPT = """You extract durable facts from a conversation. The user is
a university teacher; they may state facts about THEMSELVES or about a STUDENT.

Only extract a fact the USER explicitly stated that will still be true next
week and that does NOT live in the gradebook.

EXTRACT (example names only -- never copy a fact from these examples):
  "I teach the 8am section."   -> The user teaches the 8am section.
  "Nadia has an extended-time accommodation for exams."
                               -> Nadia has an extended-time accommodation for exams.
  "Omar emailed me -- he's been dealing with an injury."
                               -> Omar has been dealing with an injury.
  "I have a dentist appointment Tuesday so I'll cancel office hours."
                               -> The user has a dentist appointment Tuesday and
                                  is cancelling office hours that day.
  "I have an important meeting Friday so I will miss class."
                               -> The user has an important meeting Friday and
                                  will miss class.

KEEP THE REASON. If the user says WHY something is happening, that reason is
the most useful half of the fact -- never drop it. This works in BOTH
directions, and the second one is the one that gets missed:
  "I'll miss class because of a conference"  (reason second)
  "I have a conference, so I'll miss class"  (reason FIRST -- keep it anyway)
Both must keep the conference. Never reduce either to "The user will miss
class". A fact stored without its reason makes the agent invent one later.

Record what the user TOLD you, not how they told you. "Marcus emailed me, he
has a concussion" is a fact about Marcus's concussion -- not a fact about the
user receiving email.

DO NOT EXTRACT (return an empty list for all of these):
  "How is Sam doing?"          -> a question, not a fact
  "Chart the class average."   -> a request, not a fact
  "Show me how Sam compares to the class."
                               -> a request, not a fact about Sam
  "What is 12 * 40?"           -> a task, not a fact
  anything YOU said, however useful it sounded
  anything about what the user wants RIGHT NOW

Never write a fact about the user "wanting", "asking for", "being interested
in", "preparing for", or "needing" something. Those describe this moment,
not the person.

Never record data that came from a tool -- grades, averages, attendance,
due dates. It goes stale the moment the gradebook changes, and the agent can
always just call the tool again. Memory is for context ONLY the teacher could
have told you: circumstances, accommodations, preferences, history.

Write each fact as one third-person sentence -- short, but never at the cost
of the reason. Facts about the teacher start with "The user"; facts about a
student start with the student's name.
Most exchanges contain NOTHING durable. An empty list is the correct and
common answer -- prefer it when unsure.

You are also shown the assistant's reply. It is there for ONE reason: so you can
resolve what the user was pointing at when they said "them", "that", or "those".
Never extract a fact out of the assistant's reply on its own.

EXCEPTION -- if the user explicitly asks you to remember, note, or not forget
something, always extract it. Resolve what they meant from the reply and write
it out in full. "Remember them." after a list of deadlines becomes:
  The user wants to keep track of these deadlines: HW4 due 2026-09-05, ...
An explicit request overrides every other rule above."""


def _is_explicit_request(user_msg: str) -> bool:
    """Did the user literally ask us to remember something?

    ---- MODIFY HERE ----
    Deliberately a dumb keyword check, not another LLM call. It runs on every
    turn, and a wrong answer here is cheap. Save the model calls for the
    judgement that actually needs judgement.
    """
    triggers = ("remember", "don't forget", "dont forget", "note that",
                "keep track", "make a note", "save that", "memorize")
    return any(t in user_msg.lower() for t in triggers)

# ---- MODIFY HERE ----
# LiteLLM's OpenAI-compatible JSON-schema response format constrains decoding: the
# model is physically incapable of producing text that doesn't match. This is
# how we get reliable structured output from a 4B model that was never trained
# for function calling. No regex, no "please respond with valid JSON", no retry
# loop. Change the schema and the model's output shape changes with it.
_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["facts"],
}


def extract_facts(user_msg: str, assistant_msg: str = "") -> list[str]:
    """Ask the LLM what (if anything) is worth remembering from this exchange.

    ---- MODIFY HERE ----
    Facts must come from what the USER said -- feed the assistant's reply in as
    a source and memory fills up with the agent's own output ("The user has
    grades of 94/100..."), which is both stale and fetchable from a tool.

    But the reply can't be left out entirely either. "Remember them." means
    nothing on its own. So the reply goes in strictly as CONTEXT for resolving
    references, and the prompt says so. Getting this boundary right is most of
    the work in a real memory system.
    """
    explicit = _is_explicit_request(user_msg)

    # A question is never a durable fact -- and a small model, pattern-matching
    # on the examples in the prompt above, will sometimes invent one anyway
    # ("Sam has been dealing with an injury" out of "How is Sam doing?").
    # So for questions we don't even ask. The prompt already says this rule;
    # this line ENFORCES it. Prompt for the behavior you want; validate for
    # the behavior you need.
    if user_msg.strip().endswith("?") and not explicit:
        return []

    exchange = f'The user said: "{user_msg}"'

    # Only show the assistant's reply when the user said "remember this" -- that
    # is the ONLY case that needs it, to resolve what "them"/"that" points at.
    # Including it on every turn measurably pollutes memory: the model starts
    # saving its own output ("The user is attempting a multiplication problem").
    # Narrow the input to the narrow case that needs it.
    if explicit and assistant_msg:
        exchange += (
            f'\n\n(Context for resolving references only -- the assistant had '
            f'replied: "{assistant_msg}")'
            "\n\nThe user is explicitly asking you to remember something. Extract it."
        )

    response = llm.chat(
        messages=[
            {"role": "system", "content": EXTRACT_PROMPT},
            {"role": "user", "content": exchange},
        ],
        response_schema=_EXTRACT_SCHEMA,
        schema_name="memory_facts",
        temperature=0,
        max_tokens=settings.MAX_DECISION_TOKENS,
    )
    try:
        facts = json.loads(response.content).get("facts", [])
    except json.JSONDecodeError:
        return []

    # ---- MODIFY HERE ----
    # A prompt is a request, not a guarantee. Even told twice not to, a 4B model
    # will still occasionally save "The user is interested in seeing grades."
    # So we ALSO filter in code. Prompt for the behavior you want; validate for
    # the behavior you need.
    # Transient states dressed up as facts.
    JUNK = ("interested in", "wants to", "wants ", "asked", "is asking",
            "needs to know", "would like", "is curious", "requested",
            "planning to", "preparing for", "is going to", "attempting",
            "is trying to", "is calculating", "is looking for",
            "compares to", "comparison", "compared to",
            "to calculate", "to chart", "to draft",
            # NOTE: an earlier version also filtered "today"/"this friday"/etc,
            # reasoning that facts pinned to a day go stale. That was WRONG and
            # it broke a real case: "I have a meeting this Friday so I'll miss
            # class" is exactly the kind of thing a teacher needs remembered.
            # A filter that blocks junk AND the good stuff is worse than no
            # filter. Keep these lists narrow -- match on the shape of a
            # non-fact ("wants to", "is asking"), never on its subject matter.
            # Conclusions ABOUT the data, not context. The gradebook already
            # knows who is struggling; memory is for what the gradebook can't know.
            "is struggling", "is failing", "is falling behind", "below average",
            "is worried", "is concerned")

    # Tool output that will go stale (e.g. "The user has grades of 94/100...").
    STALE = ("/100", "out of 100", "due 20", "%", "score of", "scored")

    keepers = []
    for fact in facts:
        if not isinstance(fact, str):
            continue
        fact = fact.strip()
        if len(fact) < 10:
            continue
        # If the user explicitly said "remember this", we skip the filters.
        # They asked. Second-guessing them is worse than storing something
        # imperfect -- and silently saving nothing is the worst option of all.
        if not explicit and any(phrase in fact.lower() for phrase in JUNK + STALE):
            continue
        keepers.append(fact)
    return keepers


# ---------------------------------------------------------------------------
# 2. RECONCILE — the part everyone forgets
# ---------------------------------------------------------------------------
# Naive memory systems APPEND. So the teacher says "Priya has an extended-time
# accommodation", then later "Priya's accommodation ended", and now memory
# holds both. The agent gets confused and the user loses trust.
#
# Fix: before saving, check whether this fact is about the same TOPIC as
# something we already know. If so, let the model decide what to do.
RECONCILE_PROMPT = """An existing memory may conflict with a new fact.

Choose one:
  "update" - the new fact REPLACES the old one (it changed, or is more specific)
  "skip"   - the new fact adds nothing; the old one already covers it
  "add"    - they are about different things; keep both

Answer with the action only."""

_RECONCILE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["update", "skip", "add"]},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}


def _reconcile(new_fact: str, existing_fact: str) -> tuple[str, str]:
    response = llm.chat(
        messages=[
            {"role": "system", "content": RECONCILE_PROMPT},
            {"role": "user", "content": f"EXISTING: {existing_fact}\nNEW: {new_fact}"},
        ],
        response_schema=_RECONCILE_SCHEMA,
        schema_name="memory_reconciliation",
        temperature=0,
        max_tokens=settings.MAX_DECISION_TOKENS,
    )
    try:
        result = json.loads(response.content)
        return result["action"], result.get("reason", "")
    except (json.JSONDecodeError, KeyError):
        return "add", "could not parse decision, defaulting to add"


def remember(fact: str) -> str:
    """Store one fact, reconciling it against what we already know.

    Returns a human-readable description of what happened, for the trace panel.
    """
    memories = load()
    # Ask whether each existing fact is the same topic. The reconciliation
    # schema can distinguish a replacement/duplicate from an unrelated fact.
    for existing in memories:
        action, reason = _reconcile(fact, existing["text"])

        if action == "skip":
            return f"SKIP (already knew it) - {fact}"

        if action == "update":
            existing["text"] = fact
            save(memories)
            return f"UPDATE #{existing['id']} - {fact}"

    next_id = max((m["id"] for m in memories), default=0) + 1
    memories.append(
        {
            "id": next_id,
            "text": fact,
        }
    )
    save(memories)
    return f"ADD #{next_id} - {fact}"


# ---------------------------------------------------------------------------
# 3. RETRIEVE — get the relevant facts back
# ---------------------------------------------------------------------------
# Two strategies, switchable from the GUI. Run the SAME question through both.
# That side-by-side is the fastest way to compare literal and semantic search.
def retrieve(query: str, mode: str = "semantic") -> list[dict]:
    memories = load()
    if not memories:
        return []

    if mode == "keyword":
        # Naive but honest: count shared words. This is roughly what a
        # `SELECT ... WHERE text LIKE '%word%'` gets you.
        query_words = {w.strip(".,!?").lower() for w in query.split() if len(w) > 3}
        hits = []
        for m in memories:
            memory_words = {w.strip(".,!?").lower() for w in m["text"].split()}
            overlap = len(query_words & memory_words)
            if overlap > 0:
                hits.append((overlap, m))
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in hits[: settings.RETRIEVAL_TOP_K]]

    # Semantic: ask the configured DGX chat model to compare meaning. This finds
    # "vegetarian" from "what should I eat?" without an embeddings endpoint.
    selected_ids = _semantic_relevant_ids(query, memories)
    by_id = {memory["id"]: memory for memory in memories}
    return [by_id[memory_id] for memory_id in selected_ids]
