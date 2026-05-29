# Portfolio Analysis Enhancement Plan

This file tracks the work to extend TradingAgents from single-instrument analysis to whole-portfolio analysis across multiple stocks and options. The goal is to produce portfolio-level recommendations that explain whether to buy, sell, trim, add, or hold each component and how target weights should change.

## Progress Legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete

## Objectives

- [x] Accept a portfolio containing multiple stocks, options, and cash positions.
- [x] Validate each component's symbol, asset type, and current portfolio weight.
- [x] Reuse the existing single-instrument agent pipeline for each analyzable holding.
- [x] Add portfolio-level analytics for allocation, concentration, correlation, volatility, beta, and option Greeks.
- [x] Produce a final portfolio recommendation with target weights and per-component actions.
- [x] Preserve backward compatibility for the current single-ticker CLI and API flow.

## Phase 1: Portfolio Input Models

- [x] Add typed schemas for portfolio inputs.
- [x] Define `PortfolioInstrument` with fields such as `symbol`, `asset_type`, and option-specific fields.
- [x] Define `PortfolioPosition` with fields such as `instrument`, `current_weight`, `quantity`, `market_value`, `cost_basis`, and optional weight limits.
- [x] Define `PortfolioRequest` with positions, cash weight, base currency, trade date, and risk constraints.
- [x] Validate that portfolio weights sum to approximately 100%.
- [x] Validate that stock, option, and cash positions have the required fields.
- [x] Add unit tests for model validation.

Suggested locations:

- `tradingagents/portfolio/schemas.py`
- `tests/test_portfolio_models.py`

## Phase 2: Portfolio Input Parsing

- [x] Add CSV input parsing for portfolio positions.
- [x] Add JSON input parsing for portfolio positions.
- [x] Normalize ticker symbols using existing ticker handling patterns.
- [x] Support option contract fields such as underlying, expiry, strike, right, and contract symbol.
- [x] Support cash as an explicit portfolio component.
- [x] Add parser tests for valid and invalid portfolio files.

Suggested locations:

- `tradingagents/portfolio/io.py`
- `tests/test_portfolio_cli.py`

Example CSV:

```csv
symbol,asset_type,current_weight,quantity,expiry,strike,right
AAPL,stock,0.30,25,,,
MSFT,stock,0.25,12,,,
NVDA,stock,0.20,10,,,
AAPL250620C00200000,option,0.05,2,2025-06-20,200,C
CASH,cash,0.20,,,,
```

## Phase 3: Reuse Single-Instrument Analysis

- [x] Add `TradingAgentsGraph.propagate_portfolio(portfolio_request)`.
- [x] Run the existing single-instrument graph for each stock holding.
- [x] For options, analyze both the contract and the underlying where data is available.
- [x] Cache or reuse per-symbol reports within a portfolio run.
- [x] Collect each holding's analyst reports, debates, trader plan, final decision, and parsed signal.
- [x] Keep `TradingAgentsGraph.propagate(company_name, trade_date, asset_type="stock")` unchanged for existing callers.
- [x] Add tests proving single-ticker behavior remains backward compatible.

Suggested locations:

- `tradingagents/graph/trading_graph.py`
- `tests/test_portfolio_graph.py`

## Phase 4: Options Data Support

- [x] Add option contract lookup.
- [x] Add options chain retrieval.
- [x] Add implied volatility support when provided by the data vendor.
- [x] Add delta, gamma, theta, vega, and rho support when provided by the data vendor.
- [x] Add fallback Greek calculation when provider data is missing.
- [x] Compute days to expiry and moneyness.
- [x] Register options tools in the dataflow routing layer.
- [x] Add tests for option parsing, chain retrieval adapters, and Greek fallback behavior.

Suggested locations:

- `tradingagents/dataflows/yfinance_options.py`
- `tradingagents/agents/utils/options_tools.py`
- `tradingagents/dataflows/interface.py`
- `tests/test_options_dataflows.py`

## Phase 5: Portfolio Analytics

- [x] Compute current weights and market values.
- [x] Compute historical returns for each component where available.
- [x] Compute volatility for each holding.
- [x] Compute the portfolio correlation matrix.
- [x] Compute beta versus the configured benchmark.
- [x] Estimate sector, theme, or underlying concentration where data is available.
- [x] Fetch best-effort holding history, benchmark history, and sector labels before portfolio-level agent review.
- [x] Compute delta-adjusted exposure for options.
- [x] Aggregate option Greeks at the portfolio level.
- [x] Estimate risk contribution by component.
- [x] Identify overweight and underweight exposures versus configured constraints.
- [x] Add tests for each deterministic portfolio metric.

Suggested locations:

- `tradingagents/portfolio/analytics.py`
- `tradingagents/portfolio/risk.py`
- `tests/test_portfolio_analytics.py`

## Phase 6: Deterministic Rebalancing Engine

- [x] Convert single-instrument ratings into numeric recommendation scores.
- [x] Penalize high volatility, high correlation, excessive concentration, and weak liquidity.
- [x] Apply minimum and maximum component weight constraints.
- [x] Apply maximum option exposure constraints.
- [x] Apply cash and risk-budget constraints.
- [x] Produce proposed target weights before LLM review.
- [x] Generate per-component action labels: `Buy`, `Add`, `Hold`, `Trim`, `Sell`.
- [x] Add tests for rebalancing logic and constraint handling.

Suggested locations:

- `tradingagents/portfolio/rebalancing.py`
- `tests/test_portfolio_rebalancing.py`

## Phase 7: Portfolio-Level Agents

- [x] Add a portfolio risk analyst prompt or node.
- [x] Add a portfolio rebalancer prompt or node.
- [x] Add a portfolio-level manager prompt or mode.
- [x] Feed deterministic analytics and proposed target weights into the portfolio-level agents.
- [x] Ensure the LLM explains and critiques deterministic metrics instead of inventing them.
- [x] Preserve the current single-instrument portfolio manager behavior.

Suggested locations:

- `tradingagents/agents/managers/portfolio_manager.py`
- `tradingagents/agents/managers/portfolio_rebalancer.py`
- `tradingagents/agents/risk_mgmt/portfolio_risk_analyst.py`

## Phase 8: Structured Portfolio Decision Output

- [x] Add `ComponentRecommendation`.
- [x] Add a portfolio-level final decision schema.
- [x] Include current weight, target weight, weight delta, action, and rationale per component.
- [x] Include a portfolio-level action such as `Rebalance`, `Hold`, `De-risk`, or `Increase Risk`.
- [x] Include portfolio-level risk notes.
- [x] Add a renderer that writes the final decision to Markdown.
- [x] Add structured-output tests.

Candidate schema shape:

```python
class ComponentRecommendation(BaseModel):
    symbol: str
    current_weight: float
    target_weight: float
    weight_change: float
    action: Literal["Buy", "Sell", "Hold", "Trim", "Add"]
    rationale: str


class PortfolioAllocationDecision(BaseModel):
    portfolio_action: Literal["Rebalance", "Hold", "De-risk", "Increase Risk"]
    summary: str
    component_recommendations: list[ComponentRecommendation]
    risk_notes: str
```

Suggested locations:

- `tradingagents/agents/schemas.py`
- `tests/test_structured_agents.py`

## Phase 9: CLI Support

- [x] Add a CLI mode selector for single ticker versus portfolio.
- [x] Add `--portfolio-file` support for non-interactive runs.
- [x] Add interactive portfolio entry for users who do not provide a file.
- [x] Show portfolio run progress by holding and by portfolio-level stage.
- [x] Save portfolio analysis output under a portfolio-specific report directory.
- [x] Keep current `tradingagents analyze` single-ticker behavior intact.

Suggested locations:

- `cli/main.py`
- `cli/utils.py`
- `tests/test_portfolio_cli.py`

## Phase 10: Reports

- [x] Add portfolio report directory generation.
- [x] Save per-holding reports.
- [x] Save deterministic portfolio analytics as JSON.
- [x] Save portfolio risk analysis as Markdown.
- [x] Save rebalance proposal as Markdown.
- [x] Save final portfolio decision as Markdown.
- [x] Add complete report assembly for portfolio runs.

Suggested output shape:

```text
reports/portfolio_YYYYMMDD_HHMMSS/
  complete_report.md
  holdings/
    AAPL.md
    MSFT.md
    NVDA.md
  portfolio/
    analytics.json
    risk.md
    rebalance.md
    final_decision.md
```

## Phase 11: Testing And Documentation

- [x] Add focused unit tests for all portfolio modules.
- [x] Add smoke tests for a minimal two-stock portfolio.
- [x] Add tests that mock LLM and data provider calls.
- [x] Add README documentation for portfolio mode.
- [x] Add example portfolio CSV and JSON files.
- [x] Document current limitations around options data availability and vendor support.

Suggested test files:

- `tests/test_portfolio_models.py`
- `tests/test_portfolio_analytics.py`
- `tests/test_portfolio_rebalancing.py`
- `tests/test_options_dataflows.py`
- `tests/test_portfolio_cli.py`

## Implementation Order

- [x] Add portfolio schemas and validation.
- [x] Add CSV and JSON portfolio input parsing.
- [x] Add `propagate_portfolio()` that runs existing single-symbol analysis per holding.
- [x] Add deterministic portfolio analytics.
- [x] Add structured portfolio recommendation schema and renderer.
- [x] Add portfolio-level manager prompt and final report.
- [x] Add options chain and Greeks support.
- [x] Add optimizer-based rebalancing.
- [x] Wire portfolio mode into the CLI.
- [x] Expand tests and documentation.

## Open Design Decisions

- [x] Decide whether portfolio mode should be a separate CLI command or a mode inside `analyze`.
- [x] Decide the first supported option data vendor and fallback behavior.
- [x] Decide whether target weights should be absolute percentages or deltas only in the first release.
- [x] Decide how to handle unavailable data for thinly traded options.
- [ ] Decide whether portfolio memory should be stored separately from single-ticker memory.
- [x] Decide whether a portfolio run should continue when one holding fails analysis.

## Progress Notes

Use this section to record implementation milestones, decisions, and test results.

| Date | Status | Notes |
| --- | --- | --- |
| 2026-05-27 | Created | Initial portfolio analysis enhancement plan added. |
| 2026-05-27 | Phase 1 in progress | Added `tradingagents/portfolio` schemas and `tests/test_portfolio_models.py`. Attempted `pytest tests/test_portfolio_models.py`, but external process startup failed with `windows sandbox: spawn setup refresh`; verification is still pending. |
| 2026-05-27 | Phase 1 verified | Installed `pytest` in the existing `tradingagents` Conda environment and verified `conda run -n tradingagents python -m pytest tests\test_portfolio_models.py`: 14 passed. |
| 2026-05-27 | Phase 2 verified | Added CSV/JSON parsers in `tradingagents/portfolio/io.py`, exports, and `tests/test_portfolio_io.py`. Verified `conda run -n tradingagents python -m pytest tests\test_portfolio_models.py tests\test_portfolio_io.py`: 24 passed. |
| 2026-05-27 | Phase 3 verified | Added `TradingAgentsGraph.propagate_portfolio()` to orchestrate existing single-instrument analysis for stock holdings and option underlyings, cache repeated symbols, carry cash as skipped, and collect reports/debates/plans/decisions/signals per holding. Option contract analysis is marked `pending_options_data_support` until Phase 4. Verified `conda run -n tradingagents python -m pytest tests\test_portfolio_models.py tests\test_portfolio_io.py tests\test_portfolio_graph.py`: 27 passed. |
| 2026-05-27 | Phase 5 verified | Added provider-agnostic deterministic analytics in `tradingagents/portfolio/analytics.py` with allocation, market value, returns, volatility, correlation, beta, concentration, delta-adjusted exposure, aggregate Greeks, risk contribution, and constraint flags. Verified `conda run -n tradingagents python -m pytest tests\test_portfolio_models.py tests\test_portfolio_io.py tests\test_portfolio_graph.py tests\test_portfolio_analytics.py`: 31 passed. |
| 2026-05-27 | Phase 8 verified | Added structured `ComponentRecommendation` and `PortfolioAllocationDecision` schemas plus a Markdown renderer for portfolio-level final decisions. Verified `conda run -n tradingagents python -m pytest`: 266 passed. |
| 2026-05-27 | Phase 6 verified | Added deterministic rebalancing in `tradingagents/portfolio/rebalancing.py` with rating scores, risk penalties, target-weight constraints, option exposure caps, cash/risk-budget handling, action labels, and Markdown rendering. Verified `conda run -n tradingagents python -m pytest`: 277 passed. |
| 2026-05-27 | Phase 7 verified | Added portfolio risk analyst, portfolio rebalancer, and portfolio allocation manager nodes that consume deterministic analytics and rebalance proposals while keeping the existing single-instrument portfolio manager unchanged. Verified `conda run -n tradingagents python -m pytest`: 277 passed. |
| 2026-05-27 | Phase 9 verified | Added CLI portfolio mode selection, `--portfolio-file`, interactive portfolio entry, portfolio progress events, portfolio run orchestration, and portfolio-specific report output. Verified `conda run -n tradingagents python -m pytest`: 283 passed. |
| 2026-05-27 | Phase 10 verified | Extracted reusable portfolio report rendering/output to `tradingagents/portfolio/reports.py`, including default portfolio report directory names, per-holding Markdown, analytics JSON, rebalance JSON/Markdown, risk/rebalance/final Markdown, and complete report assembly. Verified `conda run -n tradingagents python -m pytest`: 287 passed. |
| 2026-05-28 | Phase 4 verified | Added yfinance option chain, contract lookup, implied-volatility, provider Greek, fallback Black-Scholes Greek, days-to-expiry, and moneyness support, plus options dataflow routing and market-node tools. Verified `uv run --with pytest python -m pytest`: 292 passed, 1 skipped. |
| 2026-05-28 | Optimizer/docs verified | Added opt-in mean-variance optimizer targets with bounds, turnover penalty, risk aversion, and diagnostics; added minimal two-stock smoke coverage, README portfolio-mode docs, and sample CSV/JSON portfolio files. Verified `uv run --with pytest python -m pytest`: 296 passed, 1 skipped. |
| 2026-05-28 | Option failure handling verified | Made option data tools return an unavailable JSON payload instead of raising when yfinance chain/Greek data fails, and made portfolio propagation record failed holdings while continuing the rest of the portfolio. Verified `uv run --with pytest python -m pytest tests/test_options_dataflows.py tests/test_portfolio_graph.py tests/test_portfolio_smoke.py`: 13 passed. |
| 2026-05-29 | Portfolio analytics input collection verified | Added best-effort collection of holding price history, configured benchmark history, and sector/industry labels before deterministic portfolio analytics. Option positions reuse underlying stock history for correlation/risk inputs, and unavailable provider data is recorded as a warning so analysis continues. Verified `uv run --with pytest python -m pytest`: 306 passed, 7 warnings. |
