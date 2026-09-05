import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mcp as external_mcp

from teacher_assistant import mcp as local_mcp
from teacher_assistant import settings
from teacher_assistant.agents import main as agent
from teacher_assistant.mcp import client as mcp_client
from teacher_assistant.mcp import course_server
from teacher_assistant.memory import store as memory
from teacher_assistant.skills import loader as skills_loader
from web_research import SearchResult, WebResearchClient


class FakeMCP:
    def __init__(self):
        self.tools = [
            SimpleNamespace(
                name="student_report",
                description="Return the complete record for one student.",
                input_schema={
                    "type": "object",
                    "properties": {"student": {"type": "string"}},
                    "required": ["student"],
                },
            ),
            SimpleNamespace(
                name="chart_grades",
                description="Render one requested grade chart.",
                input_schema={
                    "type": "object",
                    "properties": {"target": {"type": "string"}},
                },
            ),
            SimpleNamespace(
                name="calculate",
                description="Evaluate one arithmetic expression exactly.",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            ),
            SimpleNamespace(
                name=settings.WEB_RESEARCH_TOOL,
                description="Search the public web for current or unknown facts.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            ),
        ]
        self.calls = []

    def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "chart_grades":
            return (
                f"CHART_SAVED: {settings.MCP_SERVER_PATH.parent / 'charts' / 'class_averages.png'}\n"
                "The chart plots HW1: class average 82.5"
            )
        if name == "calculate":
            return "8839880"
        if name == settings.WEB_RESEARCH_TOOL:
            return (
                "Public web results for: current university policy\n"
                "1. Current answer\n"
                "URL: https://example.edu/current-answer\n"
                "Snippet: The verified current answer."
            )
        return (
            "Record for Marcus Webb:\n"
            "  - HW1: 55/100\n"
            "  - HW2: 61/100\n"
            "  - HW3: 58/100\n"
            "  - Midterm: 60/100\n"
            "  Average: 58.5/100\n"
            "  Attendance: 65%\n"
            "  Late submissions: 4"
        )


def stream_chunks(text, prompt_tokens=123):
    midpoint = max(len(text) // 2, 1)
    pieces = (text[:midpoint], text[midpoint:])
    return iter(
        [
            {"message": {"content": piece}, "done": False}
            for piece in pieces
            if piece
        ]
        + [
            {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": prompt_tokens,
            }
        ]
    )


class ChatScript:
    def __init__(self, decisions, parent_answer="Done.", specialist_answer=""):
        self.decisions = list(decisions)
        self.parent_answer = parent_answer
        self.specialist_answer = specialist_answer
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            user_prompt = kwargs["messages"][-1]["content"]
            if "Write the email now" in user_prompt:
                return stream_chunks(self.specialist_answer)
            return stream_chunks(self.parent_answer)

        decision = self.decisions.pop(0)
        return {"message": {"content": json.dumps(decision)}}


class AgentCallCountTests(unittest.TestCase):
    def run_turn(self, chat_script, text, use_subagent=True):
        mcp = FakeMCP()
        messages = []
        with (
            patch("ollama.chat", side_effect=chat_script),
            patch.object(agent.store, "retrieve", return_value=[]),
            patch.object(agent.store, "extract_facts", return_value=[]),
        ):
            events = list(
                agent.run_turn(
                    mcp,
                    messages,
                    text,
                    retrieval_mode="semantic",
                    use_subagent=use_subagent,
                )
            )
        return mcp, events

    def assert_shared_context(self, calls):
        self.assertTrue(calls)
        for call in calls:
            self.assertEqual(
                call["options"]["num_ctx"],
                settings.MODEL_CONTEXT_TOKENS,
            )

    def test_no_tool_uses_one_route_and_one_answer_call(self):
        script = ChatScript(
            [{"reasoning": "No lookup is needed.", "tool": "none", "args": {}}],
            parent_answer="Hello. How can I help?",
        )

        mcp, events = self.run_turn(script, "Hello?")

        self.assertEqual(len(script.calls), 2)
        self.assertFalse(script.calls[0].get("stream", False))
        self.assertTrue(script.calls[1]["stream"])
        self.assertEqual(
            script.calls[1]["options"]["num_predict"],
            settings.MAX_ANSWER_TOKENS,
        )
        self.assertNotIn(
            "## TOOLS YOU CAN CALL",
            script.calls[1]["messages"][0]["content"],
        )
        self.assertEqual(
            script.calls[1]["messages"][0]["content"],
            agent.PERSONA,
        )
        self.assert_shared_context(script.calls)
        self.assertEqual(mcp.calls, [])
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(events[-1][1][-1]["content"], "Hello. How can I help?")

    def test_one_tool_uses_two_routes_and_one_answer_call(self):
        script = ChatScript(
            [
                {
                    "reasoning": "The gradebook is required.",
                    "tool": "student_report",
                    "args": {"student": "Marcus Webb"},
                },
                {"reasoning": "The result is sufficient.", "tool": "none", "args": {}},
            ],
            parent_answer="Marcus has a 58.5/100 average and 65% attendance.",
        )

        mcp, events = self.run_turn(script, "How is Marcus Webb doing?")

        self.assertEqual(len(script.calls), 3)
        self.assertEqual(
            mcp.calls,
            [("student_report", {"student": "Marcus Webb"})],
        )
        self.assert_shared_context(script.calls)
        self.assertIn(
            agent.ANSWER_INSTRUCTION,
            script.calls[-1]["messages"][0]["content"],
        )
        self.assertIn("58.5/100", events[-1][1][-1]["content"])
        self.assertIn("65%", events[-1][1][-1]["content"])

    def test_unknown_current_fact_routes_to_web_research(self):
        script = ChatScript(
            [
                {
                    "reasoning": "The answer is current and needs web evidence.",
                    "tool": settings.WEB_RESEARCH_TOOL,
                    "args": {
                        "query": "current university policy",
                        "freshness": "invented-invalid-enum",
                        "limit": 999,
                    },
                },
                {"reasoning": "The search result answers it.", "tool": "none", "args": {}},
            ],
            parent_answer="The current policy is verified by the university.",
        )

        mcp, events = self.run_turn(script, "What is the current university policy?")

        self.assertEqual(
            mcp.calls,
            [
                (
                    settings.WEB_RESEARCH_TOOL,
                    {
                        "query": "current university policy",
                        "limit": settings.WEB_SEARCH_RESULT_COUNT,
                    },
                )
            ],
        )
        self.assertIn(
            "Search instead of guessing",
            script.calls[0]["messages"][-1]["content"],
        )
        self.assertIn(
            "WEB SEARCH SAFETY",
            script.calls[-1]["messages"][0]["content"],
        )
        self.assertIn("https://example.edu/current-answer", events[-1][1][-1]["content"])

    def test_delegation_streams_child_answer_without_parent_regeneration(self):
        email = (
            "Subject: Checking in\n\n"
            "Hi Marcus,\n\n"
            "I want to recognize your persistence while checking in about your "
            "58.5/100 average and 65% attendance. Please come by Thursday so we "
            "can choose one practical next step together.\n\n"
            "Best,\n[Your name]"
        )
        script = ChatScript(
            [
                {
                    "reasoning": "The email specialist owns this task.",
                    "tool": "delegate",
                    "args": {
                        "agent": "email-writer",
                        "task": "Draft a check-in email to Marcus Webb.",
                    },
                },
                {
                    "reasoning": "The student's record is required.",
                    "tool": "student_report",
                    "args": {"student": "Marcus Webb"},
                },
            ],
            specialist_answer=email,
        )

        mcp, events = self.run_turn(script, "Draft a check-in email to Marcus Webb.")

        self.assertEqual(len(script.calls), 3)
        self.assertEqual(sum(bool(c.get("stream")) for c in script.calls), 1)
        self.assertIn("Write the email now", script.calls[-1]["messages"][-1]["content"])
        self.assertNotIn(
            "## TOOLS YOU CAN CALL",
            script.calls[-1]["messages"][0]["content"],
        )
        self.assertEqual(
            mcp.calls,
            [("student_report", {"student": "Marcus Webb"})],
        )
        self.assert_shared_context(script.calls)
        self.assertEqual(events[-1][1][-1]["content"], email)
        streamed = "".join(payload for kind, payload in events if kind == "token")
        self.assertEqual(streamed, email)
        self.assertLessEqual(len(email.split()), 120)
        self.assertIn("[Your name]", email)

    def test_chart_request_renders_once_then_answers(self):
        script = ChatScript(
            [{"reasoning": "No tool selected.", "tool": "none", "args": {}}],
            parent_answer="The class average shown for HW1 is 82.5.",
        )

        mcp, events = self.run_turn(script, "Could you show me a chart of the class?")

        self.assertEqual(len(script.calls), 2)
        self.assertEqual(mcp.calls, [("chart_grades", {"target": "class"})])
        chart_events = [payload for kind, payload in events if kind == "chart"]
        expected = settings.MCP_SERVER_PATH.parent / "charts" / "class_averages.png"
        self.assertEqual(chart_events, [str(expected)])

    def test_calculation_returns_exact_tool_result_without_regeneration(self):
        script = ChatScript(
            [
                {
                    "reasoning": "Exact arithmetic is required.",
                    "tool": "calculate",
                    "args": {"expression": "4820 * 1834"},
                }
            ]
        )

        mcp, events = self.run_turn(script, "What is 4820 * 1834?")

        self.assertEqual(len(script.calls), 1)
        self.assertEqual(mcp.calls, [("calculate", {"expression": "4820 * 1834"})])
        self.assertEqual(events[-1][1][-1]["content"], "8839880")
        streamed = "".join(payload for kind, payload in events if kind == "token")
        self.assertEqual(streamed, "8839880")


class ConfigurationAndMemoryTests(unittest.TestCase):
    def test_chat_options_are_bounded_and_fresh(self):
        first = settings.chat_options(temperature=0)
        second = settings.chat_options(num_predict=12)

        first["num_ctx"] = 1
        self.assertEqual(second["num_ctx"], settings.MODEL_CONTEXT_TOKENS)
        self.assertEqual(second["num_predict"], 12)

    def test_web_research_arguments_are_bounded(self):
        oversized = " ".join(f"word{i}" for i in range(100))
        args = agent.normalize_web_research_args(
            {"query": oversized, "limit": 999},
            "fallback query",
        )

        self.assertLessEqual(len(args["query"]), 400)
        self.assertLessEqual(len(args["query"].split()), 50)
        self.assertEqual(args["limit"], settings.WEB_SEARCH_RESULT_COUNT)

    def test_semantic_retrieval_applies_the_quality_floor(self):
        records = [
            {"id": 1, "text": "relevant", "embedding": [1.0, 0.0]},
            {"id": 2, "text": "unrelated", "embedding": [0.49, 0.871722]},
        ]
        with (
            patch.object(memory, "load", return_value=records),
            patch.object(memory, "embed", return_value=[1.0, 0.0]),
            patch.object(memory, "_ensure_compatible_embeddings", return_value=records),
        ):
            recalled = memory.retrieve("query", mode="semantic")

        self.assertEqual([item["text"] for item in recalled], ["relevant"])

    def test_explicit_remember_question_still_runs_fact_extraction(self):
        response = {"message": {"content": json.dumps({"facts": ["Marcus needs extra time."]})}}
        with patch("ollama.chat", return_value=response) as chat:
            facts = memory.extract_facts("Remember that Marcus needs extra time?")

        self.assertEqual(facts, ["Marcus needs extra time."])
        self.assertEqual(
            chat.call_args.kwargs["options"]["num_ctx"],
            settings.MODEL_CONTEXT_TOKENS,
        )


class WebResearchClientTests(unittest.TestCase):
    def test_search_falls_back_to_bing_when_duckduckgo_fails(self):
        client = WebResearchClient()
        fallback = SearchResult(
            title="Fallback result",
            url="https://example.org/fallback",
            snippet="Evidence from the fallback provider.",
            query="test",
        )
        with (
            patch.object(client, "_search_ddg", side_effect=RuntimeError("blocked")),
            patch.object(client, "_search_bing", return_value=[fallback]) as bing,
        ):
            results = client.search("test", limit=3)

        bing.assert_called_once_with("test", limit=3)
        self.assertEqual(results, [fallback])

    def test_duckduckgo_redirects_are_resolved(self):
        redirect = (
            "https://duckduckgo.com/l/?uddg="
            "https%3A%2F%2Fexample.org%2Fsource%3Fa%3D1"
        )

        self.assertEqual(
            WebResearchClient._resolve_ddg_redirect(redirect),
            "https://example.org/source?a=1",
        )


class ProjectStructureTests(unittest.TestCase):
    def test_expected_python_files_are_at_the_repository_root(self):
        root_python_files = sorted(path.name for path in settings.PROJECT_ROOT.glob("*.py"))

        self.assertEqual(
            root_python_files,
            ["app.py", "setup_check.py", "web_research.py"],
        )

    def test_resource_paths_are_absolute_and_owned_by_their_subsystems(self):
        root = Path(__file__).resolve().parents[1]
        expected_mcp = root / "teacher_assistant" / "mcp"
        expected_skills = root / "teacher_assistant" / "skills"
        expected_memory = root / "teacher_assistant" / "memory" / "memories.json"

        self.assertEqual(settings.PROJECT_ROOT, root)
        self.assertEqual(settings.MCP_SERVER_PATH, expected_mcp / "course_server.py")
        self.assertEqual(settings.MEMORY_PATH, expected_memory)
        self.assertEqual(settings.SKILLS_PATH, expected_skills)
        for resource_path in (
            settings.PROJECT_ROOT,
            settings.MCP_SERVER_PATH,
            settings.MEMORY_PATH,
            settings.SKILLS_PATH,
        ):
            self.assertTrue(resource_path.is_absolute())
            self.assertEqual(resource_path, resource_path.resolve())

        self.assertEqual(memory.MEMORY_PATH, expected_memory)
        self.assertEqual(skills_loader.SKILLS_PATH, expected_skills)
        self.assertEqual(course_server.DATA_FILE, expected_mcp / "course_data.json")
        self.assertEqual(course_server.CHARTS_DIR, expected_mcp / "charts")
        self.assertTrue(course_server.DATA_FILE.is_file())

    def test_chart_generation_uses_the_mcp_runtime_directory(self):
        with tempfile.TemporaryDirectory() as scratch:
            chart_directory = Path(scratch) / "charts"
            with patch.object(course_server, "CHARTS_DIR", chart_directory):
                result = course_server.chart_grades("class")

            marker = result.splitlines()[0]
            chart_path = Path(marker.removeprefix("CHART_SAVED: "))
            self.assertEqual(chart_path.parent, chart_directory.resolve())
            self.assertTrue(chart_path.is_file())

    def test_skill_discovery_does_not_depend_on_the_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as scratch:
            try:
                os.chdir(scratch)
                skills = skills_loader.list_skills()
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            {skill["name"] for skill in skills},
            {"check-in-email", "weekly-report"},
        )
        self.assertTrue(all(skill["path"].is_absolute() for skill in skills))

    def test_mcp_client_normalizes_the_relocated_server_path(self):
        with (
            patch.object(mcp_client.asyncio, "new_event_loop") as new_event_loop,
            patch.object(mcp_client.threading, "Thread") as thread_class,
        ):
            client = mcp_client.MCPClient(settings.MCP_SERVER_PATH)

        self.assertEqual(client.server_path, settings.MCP_SERVER_PATH)
        thread_class.assert_called_once_with(
            target=new_event_loop.return_value.run_forever,
            daemon=True,
        )
        thread_class.return_value.start.assert_called_once_with()

    def test_mcp_client_defines_the_client_owned_web_tool(self):
        tool = mcp_client._web_research_tool()

        self.assertEqual(tool.name, settings.WEB_RESEARCH_TOOL)
        self.assertEqual(tool.input_schema["required"], ["query"])
        self.assertEqual(tool.input_schema["properties"]["limit"]["maximum"], 8)

    def test_mcp_client_runs_web_research_locally(self):
        with (
            patch.object(mcp_client.asyncio, "new_event_loop"),
            patch.object(mcp_client.threading, "Thread"),
        ):
            client = mcp_client.MCPClient(settings.MCP_SERVER_PATH)

        result = SearchResult(
            title="Test result",
            url="https://example.com/result",
            snippet="Useful current evidence.",
            query="test query",
        )
        with patch.object(client._web_research, "search", return_value=[result]) as search:
            output = client.call_tool(
                settings.WEB_RESEARCH_TOOL,
                {"query": "test query", "limit": 999},
            )

        search.assert_called_once_with("test query", limit=16)
        self.assertIn("Test result", output)
        self.assertIn("https://example.com/result", output)

    def test_local_mcp_package_coexists_with_the_external_sdk(self):
        self.assertEqual(external_mcp.__name__, "mcp")
        self.assertEqual(local_mcp.__name__, "teacher_assistant.mcp")
        self.assertIsNot(external_mcp, local_mcp)
        self.assertTrue(hasattr(external_mcp, "StdioServerParameters"))
        self.assertIs(mcp_client.StdioServerParameters, external_mcp.StdioServerParameters)


class FakeMCPClient:
    def __init__(self, _server_script):
        self.tools = []
        self.server_names = ["course-tools"]
        self.tool_sources = {}

    def connect(self):
        return None


class AppStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("app", None)
        with (
            patch.object(mcp_client, "MCPClient", FakeMCPClient),
            patch("ollama.embed", return_value={"embeddings": [[0.0]]}),
            patch("ollama.chat", return_value={"message": {"content": ""}}),
        ):
            cls.app = importlib.import_module("app")

    @classmethod
    def tearDownClass(cls):
        cls.app.demo.close()

    def test_first_yield_precedes_agent_work_and_token_updates_only_chat(self):
        entered_run_turn = []

        def fake_run_turn(*_args, **_kwargs):
            entered_run_turn.append(True)
            yield ("token", "Fast response")
            yield (
                "stats",
                {"prompt_tokens": 10, "skills_loaded": 0, "delegations": 0},
            )
            yield (
                "done",
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Fast response"},
                ],
            )

        with (
            patch.object(self.app, "run_turn", fake_run_turn),
            patch.object(self.app, "render_context", return_value="context") as context,
            patch.object(self.app, "render_memory", return_value="memory") as memory,
        ):
            output = self.app.on_send("Hello", [], [], "semantic", True)

            initial = next(output)
            self.assertEqual(entered_run_turn, [])
            self.assertEqual(initial[0], "")
            self.assertEqual(initial[1][-2]["content"], "Hello")
            self.assertEqual(initial[1][-1]["content"], "")
            self.assertIn("Working", initial[6])

            token = next(output)
            self.assertEqual(entered_run_turn, [True])
            self.assertEqual(token[1][-1]["content"], "Fast response")
            skip = self.app.gr.skip()
            for index, value in enumerate(token):
                if index != 1:
                    self.assertEqual(value, skip)
            context.assert_not_called()
            memory.assert_not_called()

            stats = next(output)
            self.assertIn("10 tokens", stats[6])
            context.assert_not_called()
            memory.assert_not_called()

            done = next(output)
            self.assertEqual(done[4], "context")
            self.assertEqual(done[5], "memory")
            context.assert_called_once()
            memory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
