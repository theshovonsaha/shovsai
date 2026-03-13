import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engine.core import AgentCore
from orchestration.session_manager import SessionManager
from plugins.tool_registry import ToolRegistry


class AsyncIter:
    def __init__(self, items):
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            yield item


@pytest.mark.asyncio
async def test():
    """
    Planner event behavior:
    - use_planner=True      -> emits plan when planner returns tools
    - use_planner=False     -> no plan
    - use_planner='false'   -> no plan (string bool parsing)
    """
    adapter = MagicMock()
    adapter.stream = MagicMock(side_effect=lambda *a, **k: AsyncIter(["hello"]))

    context_engine = MagicMock()
    context_engine.build_context_block.return_value = ""
    context_engine.compress_exchange = AsyncMock(return_value=("", [], []))
    context_engine.set_adapter = MagicMock()

    orchestrator = MagicMock()
    orchestrator.plan = AsyncMock(return_value=["web_search"])
    orchestrator.set_adapter = MagicMock()

    core = AgentCore(
        adapter=adapter,
        context_engine=context_engine,
        session_manager=SessionManager(),
        tool_registry=ToolRegistry(),
        orchestrator=orchestrator,
    )

    with patch("engine.core.VectorEngine") as mock_ve:
        ve = MagicMock()
        ve.query = AsyncMock(return_value=[])
        mock_ve.return_value = ve

        events_true = [ev async for ev in core.chat_stream("hello", use_planner=True)]
        assert any(ev.get("type") == "plan" for ev in events_true)

        orchestrator.plan.reset_mock()
        events_false = [ev async for ev in core.chat_stream("hello", use_planner=False)]
        assert not any(ev.get("type") == "plan" for ev in events_false)
        orchestrator.plan.assert_not_called()

        orchestrator.plan.reset_mock()
        events_false_str = [ev async for ev in core.chat_stream("hello", use_planner="false")]
        assert not any(ev.get("type") == "plan" for ev in events_false_str)
        orchestrator.plan.assert_not_called()
