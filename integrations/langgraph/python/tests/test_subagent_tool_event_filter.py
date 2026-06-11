"""Tests for filtering subagent tool call events from the SSE stream.

Background
----------
When a LangGraph agent delegates to a runtime subagent (e.g. DeepAgents'
``task`` tool spawning a research subagent), ``astream_events`` emits
events for the subagent's internal tool calls. LangGraph 1.2.3+ (PR
#7928) stamps these events with ``metadata.lc_agent_name`` so consumers
can identify them as subagent-originated.

These events are not persisted in the outer graph's checkpoint, so
emitting them as ``TOOL_CALL_START/END/RESULT`` creates a mismatch
between the SSE stream and the thread state served from
``/thread/{id}``. The fix is to suppress subagent tool call events
while leaving subagent text streaming untouched.

Test strategy
-------------
Drive ``_handle_single_event`` directly with a minimal event payload
that has ``metadata.lc_agent_name`` set, and assert that no
``TOOL_CALL_START`` is dispatched.
"""

import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk

from ag_ui.core import EventType
from ag_ui_langgraph.agent import LangGraphAgent

from tests._helpers import _record_dispatch


def _make_tool_call_event(
    lc_agent_name: str | None = None,
    tool_name: str = "web_search",
    tool_call_id: str = "functions.web_search:0",
) -> dict:
    """Build an ``on_chat_model_stream`` event with a tool_call chunk.

    Mirrors the shape LangGraph emits when a model streams a tool call.
    ``lc_agent_name`` is what langgraph 1.2.3+ adds to the metadata for
    subagent-dispatched events.
    """
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


def _make_agent_with_active_run(parent_lc_agent_name: str | None = None) -> LangGraphAgent:
    """Build a minimal LangGraphAgent with an active_run initialized.

    The agent's graph is a MagicMock — these tests never invoke
    ``astream_events``, they call ``_handle_single_event`` directly.
    """
    agent = LangGraphAgent(
        name="test",
        graph=MagicMock(),
    )
    agent.active_run = {
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
    if parent_lc_agent_name is not None:
        agent.active_run["lc_agent_name"] = parent_lc_agent_name
    _record_dispatch(agent)
    return agent


def _tool_call_start_events(dispatched):
    return [
        ev for ev in dispatched
        if getattr(ev, "type", None) == EventType.TOOL_CALL_START
    ]


class TestSubagentToolEventFilter(unittest.IsolatedAsyncioTestCase):
    """Tool call events with ``metadata.lc_agent_name`` set to a value
    different from the parent agent's ``lc_agent_name`` should be
    suppressed.
    """

    async def test_outer_agent_tool_event_is_emitted(self):
        """Sanity: a tool call from the parent agent is emitted as
        ``TOOL_CALL_START``. This confirms the test harness works
        before we assert the suppression below."""
        agent = _make_agent_with_active_run(parent_lc_agent_name="manuel")
        event = _make_tool_call_event(lc_agent_name=None)  # no subagent tag

        async def _drive():
            out = []
            async for ev in agent._handle_single_event(event, state={}):
                out.append(ev)
            return out

        await _drive()
        starts = _tool_call_start_events(agent.dispatched)
        self.assertEqual(
            len(starts), 1,
            f"expected 1 TOOL_CALL_START from parent, got {len(starts)}",
        )

    async def test_subagent_tool_event_is_suppressed(self):
        """A tool call event whose ``lc_agent_name`` differs from the
        parent agent's must NOT emit ``TOOL_CALL_START``."""
        agent = _make_agent_with_active_run(parent_lc_agent_name="manuel")
        event = _make_tool_call_event(lc_agent_name="research")

        async def _drive():
            out = []
            async for ev in agent._handle_single_event(event, state={}):
                out.append(ev)
            return out

        await _drive()
        starts = _tool_call_start_events(agent.dispatched)
        self.assertEqual(
            len(starts), 0,
            f"subagent tool event leaked: {len(starts)} TOOL_CALL_START "
            f"dispatched; expected 0. Dispatched: "
            f"{[type(e).__name__ for e in agent.dispatched]}",
        )

    async def test_same_named_subagent_is_not_suppressed(self):
        """A subagent with the same ``lc_agent_name`` as the parent
        (self-invocation or same-named nested graph) should NOT be
        filtered — that matches the langgraph 1.2.3+ semantics where
        the discriminator is "lc_agent_name present", not "differs
        from parent"."""
        agent = _make_agent_with_active_run(parent_lc_agent_name="manuel")
        event = _make_tool_call_event(lc_agent_name="manuel")

        async def _drive():
            out = []
            async for ev in agent._handle_single_event(event, state={}):
                out.append(ev)
            return out

        await _drive()
        starts = _tool_call_start_events(agent.dispatched)
        self.assertEqual(
            len(starts), 1,
            f"same-named subagent should pass through; got {len(starts)}",
        )


if __name__ == "__main__":
    unittest.main()
