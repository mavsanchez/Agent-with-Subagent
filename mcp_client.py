"""
mcp_client.py — Plumbing. Connects our agent to the MCP server.

You can skim this file. It is the least interesting part of the demo -- it
exists purely to bridge two worlds:

  - The MCP SDK is ASYNC (async/await).
  - Gradio event handlers and our agent loop are SYNC (plain functions).

So we spin up a background thread that owns an asyncio event loop, keep the
MCP connection alive inside it, and hand out two ordinary blocking methods:

    client.tools              -> list of tools the server advertises
    client.call_tool(n, args) -> str

WHAT'S WORTH NOTICING
---------------------
The agent asks the server what tools exist (`list_tools`) instead of being
told at build time. Everything downstream -- the system prompt, the JSON
schema that constrains tool choice, the GUI panel -- is generated from that
one call.
"""

import asyncio
import sys
import threading
from contextlib import AsyncExitStack

from mcp import StdioServerParameters, stdio_client
from mcp.client import Client


class MCPClient:
    """A synchronous wrapper around one MCP server connection."""

    def __init__(self, server_script: str):
        self.server_script = server_script
        self.tools = []  # populated by connect()

        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None

        # Background thread running its own event loop, forever.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    # -- public, synchronous API --------------------------------------------

    def connect(self) -> None:
        """Launch the server subprocess and ask it what it can do."""
        self._run(self._connect())

    def call_tool(self, name: str, args: dict) -> str:
        """Call a tool on the server and return its text output."""
        result = self._run(self._client.call_tool(name, args))
        # MCP tools return a list of content blocks (text, images, ...).
        # Our tools only ever return text, so we just join the text parts.
        return "\n".join(c.text for c in result.content if hasattr(c, "text"))

    def close(self) -> None:
        try:
            self._run(self._stack.aclose())
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- internals ----------------------------------------------------------

    def _run(self, coro, timeout: int = 60):
        """Run a coroutine on the background loop and block until it's done."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    async def _connect(self) -> None:
        # sys.executable = the python running right now, so the server lands in
        # the same conda/venv environment. Avoids "works on my machine" pain.
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.server_script],
        )

        # stdio transport: we LAUNCH the server as a subprocess and talk to it
        # over its stdin/stdout. Swap in a URL string here instead and you're
        # talking to a remote server over HTTP -- nothing else changes.
        self._stack = AsyncExitStack()
        self._client = await self._stack.enter_async_context(Client(stdio_client(params)))

        # <<< THE IMPORTANT LINE >>>
        # We ask the server what tools it has. We never hardcode them.
        self.tools = (await self._client.list_tools()).tools
