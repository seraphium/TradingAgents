"""Deterministic end-to-end coverage for the single-ticker analyze workflow."""

import json
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


class DeterministicWorkflowLLM(BaseChatModel):
    """A no-network chat model that returns one response per workflow node."""

    responses: list[str]
    invocations: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "deterministic-workflow-test"

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "DeterministicWorkflowLLM":
        # Returning messages without tool calls makes every analyst complete in
        # one pass while still exercising the graph's analyst routing logic.
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        # Exercise the production fallback used by providers that do not offer
        # structured output, without involving a real model.
        raise NotImplementedError("structured output is intentionally disabled")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.responses:
            raise AssertionError("analyze workflow made more LLM calls than expected")
        self.invocations.append(messages)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.responses.pop(0)))]
        )


@pytest.mark.smoke
def test_analyze_workflow_runs_end_to_end_without_real_llm_or_market_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run every analyze phase and verify its final persisted decision."""
    expected_responses = [
        "deterministic market report",
        "deterministic sentiment report",
        "deterministic news report",
        "deterministic fundamentals report",
        "deterministic bull case",
        "deterministic bear case",
        "deterministic research plan",
        "deterministic trader proposal",
        "deterministic aggressive risk view",
        "deterministic conservative risk view",
        "deterministic neutral risk view",
        "**Rating**: Buy\n\nDeterministic final portfolio decision.",
    ]
    fake_llm = DeterministicWorkflowLLM(responses=expected_responses.copy())
    fake_client = MagicMock()
    fake_client.get_llm.return_value = fake_llm

    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.create_llm_client",
        lambda **kwargs: fake_client,
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.get_news.func",
        lambda *args, **kwargs: "deterministic headline data",
    )
    monkeypatch.setattr(
        "tradingagents.agents.analysts.sentiment_analyst.fetch_stocktwits_messages",
        lambda *args, **kwargs: "deterministic StockTwits data",
    )

    config = {
        **DEFAULT_CONFIG,
        "results_dir": str(tmp_path / "results"),
        "data_cache_dir": str(tmp_path / "cache"),
        "memory_log_path": str(tmp_path / "memory" / "trading_memory.md"),
        "checkpoint_enabled": False,
        "enable_reddit_data": False,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }

    graph = TradingAgentsGraph(config=config)
    final_state, signal = graph.propagate("DEMO", "2026-06-10")

    assert signal == "Buy"
    assert final_state["company_of_interest"] == "DEMO"
    assert final_state["trade_date"] == "2026-06-10"
    assert final_state["market_report"] == expected_responses[0]
    assert final_state["sentiment_report"] == expected_responses[1]
    assert final_state["news_report"] == expected_responses[2]
    assert final_state["fundamentals_report"] == expected_responses[3]
    assert final_state["investment_debate_state"]["count"] == 2
    assert final_state["risk_debate_state"]["count"] == 3
    assert final_state["final_trade_decision"] == expected_responses[-1]
    assert len(fake_llm.invocations) == len(expected_responses)
    assert fake_llm.responses == []

    state_log = (
        tmp_path
        / "results"
        / "DEMO"
        / "TradingAgentsStrategy_logs"
        / "full_states_log_2026-06-10.json"
    )
    persisted_state = json.loads(state_log.read_text(encoding="utf-8"))
    assert persisted_state["final_trade_decision"] == expected_responses[-1]
    assert persisted_state["trader_investment_decision"] == expected_responses[7]

    memory_log = (tmp_path / "memory" / "trading_memory.md").read_text(
        encoding="utf-8"
    )
    assert "[2026-06-10 | DEMO | Buy | pending]" in memory_log
    assert expected_responses[-1] in memory_log
