"""
client.py — Connects the agent to MCP plus key-free web research.

The MCP SDK is async while Gradio and the teaching agent are synchronous. A
background event loop keeps the course MCP connection alive and exposes two
ordinary blocking attributes:

    client.tools              -> MCP tools plus client-owned web_research
    client.call_tool(n, args) -> text returned by either capability

The web tool deliberately shares this interface. The agent loop does not need
to know whether a capability came from an MCP subprocess or lives in the
client; it discovers a schema, chooses a name, and calls it in exactly the same
way.
"""

import asyncio
import json
import sys
import threading
from contextlib import AsyncExitStack
from os import PathLike
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from mcp.client import Client
from mcp_types import Tool

from teacher_assistant import settings
from web_research import WebResearchClient, deduplicate_results, format_results


WEB_RESEARCH_DESCRIPTION = """Search the public web without an API key.

Use this when a factual answer may have changed, the user asks for current or
latest information, or you are not confident you know the answer. It searches
DuckDuckGo and falls back to Bing, returning result titles, snippets, and URLs.
Do not use it for private student facts, the teacher's personal information,
arithmetic, writing requests, opinions, or facts already present in context.
"""


def _web_research_tool() -> Tool:
    """Describe the client-owned tool in the same shape as discovered MCP tools."""
    return Tool(
        name=settings.WEB_RESEARCH_TOOL,
        description=WEB_RESEARCH_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused public-web search query.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "default": settings.WEB_SEARCH_RESULT_COUNT,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


class MCPClient:
    """A synchronous facade over course MCP tools and local web research."""

    def __init__(self, server_path: str | PathLike[str]):
        self.server_path = Path(server_path).expanduser().resolve()
        self.tools = []  # populated by connect()
        self.server_names = ["course-tools"]
        self.tool_sources: dict[str, str] = {}

        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._web_research = WebResearchClient(
            timeout=settings.WEB_RESEARCH_TIMEOUT_SECONDS
        )

        # Background thread running its own event loop, forever.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    # -- public, synchronous API --------------------------------------------

    def connect(self) -> None:
        """Launch the course MCP server and assemble the complete tool menu."""
        self._run(self._connect())

    def call_tool(self, name: str, args: dict) -> str:
        """Call either client-owned web research or a discovered MCP tool."""
        if name == settings.WEB_RESEARCH_TOOL:
            return self._call_web_research(args)

        if self._client is None or name not in self.tool_sources:
            available = ", ".join(sorted(self.tool_sources)) or "none"
            raise ValueError(f"Unknown tool {name!r}. Available tools: {available}")

        result = self._run(self._client.call_tool(name, args))
        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
        if text:
            return text
        structured = getattr(result, "structured_content", None)
        return json.dumps(structured, ensure_ascii=False) if structured is not None else ""

    def close(self) -> None:
        try:
            if self._stack is not None:
                self._run(self._stack.aclose())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- internals ----------------------------------------------------------

    def _call_web_research(self, args: dict) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValueError("web_research requires a non-empty query")
        query = " ".join(query.split()[:50])[:400].strip()

        # Keep requests and returned context bounded even if the model invents
        # a huge or malformed optional limit.
        try:
            limit = int(args.get("limit", settings.WEB_SEARCH_RESULT_COUNT))
        except (TypeError, ValueError):
            limit = settings.WEB_SEARCH_RESULT_COUNT
        limit = max(1, min(limit, 8))

        candidates = self._web_research.search(query, limit=min(limit * 2, 16))
        results = deduplicate_results(candidates, keep=limit)
        return format_results(query, results)

    def _run(self, coro, timeout: int = 60):
        """Run a coroutine on the background loop and block until it is done."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self) -> None:
        # sys.executable keeps the subprocess in this UV-managed environment.
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_path)],
        )
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(stdio_client(params)))

        discovered = (await self._client.list_tools()).tools
        web_tool = _web_research_tool()
        if any(tool.name == web_tool.name for tool in discovered):
            await self._stack.aclose()
            self._stack = None
            self._client = None
            raise RuntimeError(
                f"MCP tool name collides with client capability {web_tool.name!r}"
            )

        self.tools = [*discovered, web_tool]
        self.tool_sources = {
            **{tool.name: "course-tools" for tool in discovered},
            web_tool.name: "client",
        }
