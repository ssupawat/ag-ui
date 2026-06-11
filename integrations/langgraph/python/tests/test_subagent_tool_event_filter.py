"""Tests for filtering subagent tool call events.

Langgraph 1.2.3+ (PR #7928) stamps events from runtime-dispatched
subagents with ``metadata.lc_agent_name``. Suppress tool call events
when this signal differs from the parent agent's name.
"""

import unittest

from langchain_core.messages import AIMessageChunk

from ag_ui.core import EventType

from tests._helpers import _record_dispatch, make_agent


def _make_tool_call_event(
    lc_agent_name: str | None = None,
    tool_name: str = "web_search",
    tool_call_id: str = "functions.web_search:0",
) -> dict:
    chunk = AIMessageChunk(
        content="",
        id="lc_run--test",
        tool_calls=[{
            "name": tool_name,
            "args": {"query": "test"},
            "id": tool_call_id,
            "type": "tool_call",
        }],
        tool_call_chunks=[{
            "name": tool_name,
            "args": '{"query": "test"}',
            "id": tool_call_id,
            "index": 0,
            "type": "tool_call_chunk",
        }],
    )
    metadata = {"langgraph_node": "model", "ls_integration": "langchain_chat_model"}
    if lc_agent_name is not None:
        metadata["lc_agent_name"] = lc_agent_name
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": chunk},
        "metadata": metadata,
        "run_id": "test-run",
    }


def _active_run(lc_agent_name: str | None = None) -> dict:
    run = {
        "id": "run-1",
        "thread_id": "thread-1",
        "mode": "start",
        "reasoning_process": None,
        "node_name": None,
        "has_function_streaming": False,
        "model_made_tool_call": False,
        "state_reliable": True,
        "streamed_tool_call_ids": set(),
        "schema_keys": {
            "input": ["messages"],
            "output": ["messages"],
            "config": [],
            "context": [],
        },
    }
    if lc_agent_name is not None:
        run["lc_agent_name"] = lc_agent_name
    return run


def _tool_call_starts(dispatched):
    return [
        ev for ev in dispatched
        if getattr(ev, "type", None) == EventType.TOOL_CALL_START
    ]


class TestSubagentToolEventFilter(unittest.IsolatedAsyncioTestCase):
    async def _drive(self, parent_lc_agent_name, event):
        agent = make_agent()
        agent.active_run = _active_run(parent_lc_agent_name)
        _record_dispatch(agent)
        async for _ in agent._handle_single_event(event, state={}):
            pass
        return agent.dispatched

    async def test_outer_agent_tool_event_is_emitted(self):
        event = _make_tool_call_event(lc_agent_name=None)
        dispatched = await self._drive("manuel", event)
        self.assertEqual(len(_tool_call_starts(dispatched)), 1)

    async def test_subagent_tool_event_is_suppressed(self):
        event = _make_tool_call_event(lc_agent_name="research")
        dispatched = await self._drive("manuel", event)
        self.assertEqual(len(_tool_call_starts(dispatched)), 0)

    async def test_same_named_subagent_is_not_suppressed(self):
        event = _make_tool_call_event(lc_agent_name="manuel")
        dispatched = await self._drive("manuel", event)
        self.assertEqual(len(_tool_call_starts(dispatched)), 1)

    async def test_subagent_tool_end_is_suppressed(self):
        from langchain_core.messages import ToolMessage

        tool_msg = ToolMessage(
            content="tool output",
            tool_call_id="functions.web_search:0",
            name="web_search",
        )
        event = {
            "event": "on_tool_end",
            "data": {"output": tool_msg},
            "metadata": {
                "langgraph_node": "tools",
                "lc_agent_name": "research",
            },
            "run_id": "test-run",
        }
        agent = make_agent()
        agent.active_run = _active_run("manuel")
        _record_dispatch(agent)
        async for _ in agent._handle_single_event(event, state={}):
            pass
        result_events = [
            ev for ev in agent.dispatched
            if getattr(ev, "type", None) == EventType.TOOL_CALL_RESULT
        ]
        self.assertEqual(len(result_events), 0)


if __name__ == "__main__":
    unittest.main()
