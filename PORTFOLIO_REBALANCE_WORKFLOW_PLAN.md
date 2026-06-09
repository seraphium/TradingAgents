# Portfolio Rebalance Agent Workflow Optimization Plan

## 1. Purpose

This plan defines the next-generation portfolio rebalance workflow for TradingAgents. The target is not merely to place several single-instrument reports next to a portfolio report. The target is to produce a final, traceable, and numerically valid rebalance recommendation by combining:

- each instrument's independent analysis and initial `Buy` / `Overweight` / `Hold` / `Underweight` / `Sell` proposal;
- current instrument weights and user-supplied allocation constraints;
- sector, industry, theme, underlying, and asset-type exposure;
- portfolio volatility, correlation, beta, risk contribution, concentration, cash, and option risk;
- the current broad-market trend, benchmark trend, market regime, and recent news;
- deterministic allocation and constraint enforcement;
- portfolio-level risk review and final approval;
- a clear report containing instrument summaries, market context, rebalance rationale, and the final trade list.

The central design principle is:

> LLM agents analyze, explain, challenge, and recommend constraint changes. Deterministic code calculates and validates final target weights.

This prevents a final LLM response from silently replacing valid optimizer output with unvalidated weights.

---

## 2. Desired User Outcome

For a submitted portfolio, the user should receive one authoritative result that answers:

1. What is the current broad-market regime and benchmark trend?
2. What is the initial view on each instrument, and why?
3. What are the portfolio's main concentration and risk issues?
4. How do portfolio-level conditions modify the instrument-level proposals?
5. What are the final target weights and required trades?
6. Why is each final action different from, or consistent with, the initial instrument view?
7. Which data was missing, which constraints were applied, and how reliable is the result?

The default report should make the final decision obvious without requiring the user to compare several repetitive agent narratives.

---

## 3. Current-State Assessment

### 3.1 Capabilities already available

The current implementation already provides a strong foundation:

- portfolio request models, CSV/JSON parsing, and allocation constraints;
- per-holding reuse of the existing single-instrument analysis graph;
- per-instrument final ratings that can be converted into rebalance scores;
- provider-derived price history, benchmark data, and sector labels;
- deterministic analytics for volatility, correlation, beta, risk contribution, exposure, options, and constraint flags;
- heuristic and mean-variance rebalance modes;
- portfolio-wide market context containing global news, benchmark news, benchmark fundamentals, and benchmark indicators;
- portfolio risk, rebalance review, and final allocation agents;
- per-holding reports, portfolio reports, and CLI progress output.

### 3.2 Main gaps

The current workflow is close to the desired user experience, but several gaps prevent it from reliably producing the intended final allocation:

1. **Portfolio orchestration lives in the CLI.** Business workflow, progress display, agent invocation, report prompts, and persistence are mixed together.
2. **Market context is explanatory, not allocative.** Broad-market trend and recent news are passed to portfolio agents, but they do not directly and deterministically change optimizer constraints or risk budgets.
3. **Portfolio-level agent responsibilities overlap.** Risk Analyst, Rebalancer, and Allocation Manager repeatedly analyze similar inputs without a strict decision protocol.
4. **The Rebalancer is currently a reviewer.** It returns Markdown objections but cannot request a deterministic re-optimization.
5. **The final LLM decision can repeat or invent weights.** There is no portfolio-level validator proving that its component weights exactly match the deterministic proposal.
6. **There is no typed portfolio workflow state.** The workflow passes a growing dictionary with loosely defined fields.
7. **Missing-data behavior is permissive.** The run continues on failed holdings and unavailable metrics without a formal data-quality gate controlling whether risk may be increased.
8. **Instrument proposals are too shallow.** A five-tier rating is useful, but portfolio construction also needs conviction, horizon, risk flags, and data confidence.
9. **Options use underlying analysis as a fallback.** Underlying conviction alone is insufficient to justify increasing a specific option contract.
10. **The final report is repetitive.** It does not clearly separate initial instrument views, market overlay, final approved weights, and supporting detail.

---

## 4. Target End-to-End Workflow

```text
Portfolio Request
    |
    v
1. Input & Constraint Gate
    |
    v
2. Instrument Analysis Fan-out
    |
    v
3. Instrument Proposal Normalization
    |
    +--------------------------+
    |                          |
    v                          v
4. Portfolio Data Assembly     5. Market Regime Analysis
    |                          |
    +-------------+------------+
                  |
                  v
6. Data Quality Gate
    |
    +-- insufficient ------> Hold-only / De-risk result with warnings
    |
    v
7. Deterministic Portfolio Analytics
    |
    v
8. Initial Deterministic Rebalance Proposal (v1)
    |
    v
9. Portfolio Risk Controller
    |
    v
10. Allocation Proposal Reviewer
    |
    +-- rerun requested --> Constraint Overlay --> Re-optimize (v2, bounded loop)
    |
    v
11. Portfolio Decision Approver
    |
    v
12. Final Result Validation
    |
    v
13. Decision-First Report & Persistence
```

The workflow must produce a single authoritative `FinalRebalanceResult`. Agent reviews are supporting evidence, not competing final answers.

---

## 5. Step-by-Step Workflow Specification

## Step 1: Input and Constraint Gate

### Purpose

Validate the submitted portfolio before spending time or LLM tokens on analysis.

### Inputs

- `PortfolioRequest`;
- current weights, quantities, market values, and asset types;
- user constraints such as min/max cash, max position weight, max option weight, and per-position bounds;
- optional custom sector, industry, theme, benchmark, liquidity, and optimizer settings.

### Actions

- validate that weights sum to 100% within tolerance;
- validate symbols, option fields, and unique component identifiers;
- validate that allocation bounds are feasible together;
- normalize benchmark and classification mappings;
- assign a stable `portfolio_run_id`;
- record the effective configuration and constraints used for the run.

### Outputs

- validated `PortfolioRequest`;
- `EffectivePortfolioConstraints`;
- input warnings and blocking validation errors.

### Failure policy

Blocking validation errors stop the workflow before instrument analysis. Non-blocking warnings are retained in workflow state and the final report.

---

## Step 2: Instrument Analysis Fan-out

### Purpose

Independently analyze each analyzable instrument and obtain the evidence needed for an initial instrument-level proposal.

### Inputs

- validated portfolio positions;
- analysis date;
- selected analysts and model configuration;
- previous instrument memory, when enabled.

### Actions

For each stock or supported instrument:

- run market, sentiment, news, and fundamentals analysis;
- run bull/bear research debate;
- create the research and trader plans;
- run instrument-level risk debate;
- generate the final instrument decision.

Cash is carried through as a funding and liquidity component. Options must record whether analysis is based only on the underlying or includes contract-specific evidence.

### Outputs

One `HoldingAnalysisResult` per component, including:

- analysis status;
- analyst summaries;
- research and trader conclusions;
- instrument-level risk conclusion;
- raw final instrument decision;
- missing-data and failure details.

### Execution policy

- cache repeated underlying analysis;
- allow bounded concurrency for independent holdings;
- continue after non-critical holding failures;
- preserve explicit failure status rather than silently converting failure to `Hold`.

---

## Step 3: Instrument Proposal Normalization

### Purpose

Convert each instrument's narrative decision into a stable, structured input for portfolio construction.

### Required output schema

```python
InstrumentProposal(
    symbol: str,
    rating: Buy | Overweight | Hold | Underweight | Sell,
    score: float,
    conviction: float,
    horizon: str,
    summary: str,
    positive_drivers: list[str],
    key_risks: list[str],
    invalidation_conditions: list[str],
    data_confidence: float,
    analysis_status: str,
)
```

### Actions

- consume structured instrument decisions directly when available;
- retain text parsing only as a compatibility fallback;
- normalize rating, score, conviction, and horizon;
- reduce conviction when analysis coverage is incomplete;
- prohibit failed analysis from being interpreted as a positive proposal.

### Outputs

- `instrument_proposals_by_symbol`;
- instrument-analysis coverage metrics;
- proposal-normalization warnings.

---

## Step 4: Portfolio Data Assembly

### Purpose

Collect the deterministic inputs needed to understand the portfolio as a whole.

### Inputs to collect

- historical prices for holdings and underlyings;
- benchmark price history;
- sector and, where available, industry classification;
- user-supplied theme and custom classification mappings;
- option chain, Greeks, implied volatility, and liquidity where supported;
- position liquidity estimates;
- current market value and cost-basis information where available.

### Outputs

- `PortfolioAnalyticsInputs`;
- data-source provenance;
- missing-data warnings;
- coverage by metric and by portfolio weight.

### Important rule

Provider failure should not automatically stop the run, but every unavailable metric must be visible to the Data Quality Gate.

---

## Step 5: Market Regime Analysis

### Purpose

Convert broad-market data and recent news into a structured market overlay that can influence portfolio construction.

### Inputs

- global market news;
- benchmark news and fundamentals;
- benchmark technical indicators and recent price trend;
- optional volatility index, rates, macro, or credit data when providers support them;
- analysis date and configured lookback window.

### Required output schema

```python
MarketRegime(
    benchmark_symbol: str,
    trend: bullish | neutral | bearish,
    risk_regime: risk_on | mixed | risk_off,
    volatility_regime: low | normal | high,
    news_sentiment: positive | mixed | negative,
    confidence: float,
    equity_risk_multiplier: float,
    recommended_min_cash_weight: float | None,
    sector_overlays: dict[str, float],
    summary: str,
    evidence: list[str],
    warnings: list[str],
)
```

### Actions

- deterministically summarize benchmark trend indicators where possible;
- use an LLM agent to interpret news and market context into the structured schema;
- validate numeric overlay ranges;
- attach confidence and warnings when context is incomplete.

### Outputs

- structured `MarketRegime`;
- concise market-trend summary for the final report;
- proposed risk-budget and sector overlays for deterministic validation.

### Important rule

The market-regime agent does not output final target weights. It outputs bounded overlays and rationale.

---

## Step 6: Data Quality Gate

### Purpose

Decide whether the available evidence is sufficient for a normal rebalance, a restricted rebalance, or no risk increase.

### Inputs

- holding-analysis statuses and weighted coverage;
- instrument proposal confidence;
- market-regime confidence;
- analytics-input coverage;
- missing volatility, correlation, benchmark, sector, liquidity, and option data.

### Example policies

- if failed holding-analysis weight exceeds a threshold, prohibit `Increase Risk`;
- a failed or unsupported holding may not receive a higher target weight;
- an option without adequate contract-level data may not receive a higher target weight;
- if portfolio volatility cannot be computed, disable volatility-driven risk increases;
- if benchmark context is unavailable, reduce market-overlay confidence;
- if effective evidence coverage is below the configured minimum, produce a Hold-only or De-risk result.

### Outputs

```python
DataQualityAssessment(
    status: pass | restricted | insufficient,
    weighted_analysis_coverage: float,
    metric_coverage: dict[str, float],
    restrictions: list[str],
    blocking_issues: list[str],
    warnings: list[str],
)
```

---

## Step 7: Deterministic Portfolio Analytics

### Purpose

Compute the portfolio-level facts used by the optimizer and risk controller.

### Required analytics

- current and computed weights;
- market values;
- volatility by symbol;
- correlation matrix;
- beta by symbol;
- portfolio volatility;
- risk contribution by symbol;
- asset-type, underlying, sector, industry, and theme exposure where available;
- cash exposure;
- option exposure and aggregate Greeks;
- liquidity observations;
- deterministic constraint flags.

### Outputs

- `PortfolioAnalytics`;
- analytics diagnostics and warnings;
- a concise portfolio-risk summary for the final report.

### Important rule

Unavailable metrics remain explicit. They must not be fabricated by an LLM.

---

## Step 8: Initial Deterministic Rebalance Proposal

### Purpose

Generate proposal version 1 using instrument views, current weights, portfolio analytics, constraints, data-quality restrictions, and validated market overlays.

### Inputs

- `InstrumentProposal` objects;
- current portfolio weights;
- `PortfolioAnalytics`;
- effective user constraints;
- `DataQualityAssessment` restrictions;
- validated `MarketRegime` overlays.

### Actions

- convert instrument rating and conviction into expected-return or allocation scores;
- apply market-regime risk multiplier and cash overlay;
- apply sector/industry/theme overlays only within configured bounds;
- apply data-quality restrictions;
- run the selected heuristic or mean-variance optimizer;
- enforce component, cash, options, concentration, turnover, and exposure constraints;
- normalize and validate target weights.

### Required output

```python
RebalanceProposal(
    proposal_id: str,
    version: int,
    inputs_hash: str,
    portfolio_action: str,
    component_proposals: list[RebalanceComponent],
    target_weights_by_symbol: dict[str, float],
    risk_notes: list[str],
    rebalance_reason: str,
    diagnostics: dict[str, Any],
)
```

### Important rule

This is the first point at which target weights are generated. Target weights are always generated and validated by deterministic code.

---

## Step 9: Portfolio Risk Controller

### Purpose

Determine whether the proposed portfolio is acceptable from a whole-portfolio risk and constraint perspective.

### Inputs

- current portfolio analytics;
- proposal v1;
- market regime;
- data-quality assessment;
- instrument-level risk evidence;
- configured risk and allocation limits.

### Responsibilities

- identify hard constraint violations;
- identify excessive concentration, correlation, volatility, beta, liquidity, option, or cash risk;
- identify conflicts between market regime and proposed aggregate risk;
- identify whether missing data makes the proposal unsafe;
- recommend bounded constraint changes when needed.

### Required output

```python
RiskValidationResult(
    status: pass | warning | fail,
    violations: list[str],
    warnings: list[str],
    recommended_constraint_changes: list[ConstraintAdjustment],
    rationale: str,
)
```

### Important rule

The Risk Controller cannot directly edit target weights.

---

## Step 10: Allocation Proposal Reviewer and Bounded Re-optimization

### Purpose

Review whether the proposal appropriately reconciles instrument conviction with portfolio-level allocation trade-offs.

### Inputs

- proposal;
- instrument proposals;
- market regime;
- portfolio analytics;
- risk validation result.

### Responsibilities

- verify that strongest positive and negative instrument views are reflected appropriately;
- check whether risk constraints caused counterintuitive outcomes that require explanation;
- assess turnover and diversification trade-offs;
- choose one explicit decision:
  - `approve`;
  - `approve_with_warnings`;
  - `reject`;
  - `rerun_with_constraints`.

### Required output

```python
ProposalReview(
    decision: approve | approve_with_warnings | reject | rerun_with_constraints,
    proposal_id: str,
    objections: list[str],
    requested_constraint_changes: list[ConstraintAdjustment],
    rationale: str,
)
```

### Re-optimization loop

When the review or risk controller requests a rerun:

1. validate requested constraint changes against configured limits;
2. record why proposal v1 was rejected;
3. generate proposal v2 with deterministic code;
4. rerun final risk validation;
5. stop after the configured maximum number of reruns, initially one.

Every proposal version must remain available for diagnostics and reporting.

---

## Step 11: Portfolio Decision Approver

### Purpose

Approve or reject the final immutable proposal and provide a concise whole-portfolio explanation.

### Inputs

- final proposal version;
- final risk validation;
- proposal review;
- market regime summary;
- data-quality assessment.

### Required output

```python
PortfolioApprovalDecision(
    decision: approve | approve_with_warnings | reject,
    approved_proposal_id: str | None,
    portfolio_action: str,
    summary: str,
    key_reasons: list[str],
    execution_notes: list[str],
    warnings: list[str],
)
```

### Important rules

- the Approver does not generate or modify target weights;
- approval must reference an existing immutable proposal ID;
- rejection produces no executable trade list;
- the final narrative must explain how market regime, portfolio risk, and instrument views affected the decision.

---

## Step 12: Final Result Validation

### Purpose

Guarantee that the output is internally consistent before presentation or persistence.

### Required checks

- every input component appears exactly once;
- no unknown component appears;
- target weights sum to 100% within tolerance;
- current weights match the validated request;
- target weights match the approved deterministic proposal;
- weight changes equal target minus current;
- action labels match weight changes;
- all bounds and hard constraints are satisfied;
- proposal ID and version references are valid;
- rejected decisions contain no executable final trade list;
- all missing-data restrictions were respected.

### Outputs

- validated `FinalRebalanceResult`;
- blocking validation errors if any invariant fails.

---

## Step 13: Decision-First Report and Persistence

### Purpose

Present one clear answer while preserving detailed evidence for audit and debugging.

### Default report order

1. **Executive Decision**
   - approval status;
   - portfolio action;
   - market regime;
   - data-confidence status;
   - concise portfolio-level summary.

2. **Final Rebalance Table**

   ```text
   Instrument | Initial Rating | Current | Target | Change | Action | Final Reason
   ```

3. **Current Market Trend and News Impact**
   - benchmark trend;
   - risk regime;
   - relevant recent-news themes;
   - allocation implications.

4. **Portfolio Risk and Exposure Summary**
   - concentration;
   - sector/industry/theme exposure;
   - volatility, correlation, beta, and risk contribution;
   - cash and options exposure;
   - constraints and warnings.

5. **Instrument Analysis Summaries**
   - initial proposal;
   - conviction and key evidence;
   - final portfolio-level action;
   - explanation of any difference between the initial view and final target.

6. **Rebalance Rationale and Proposal History**
   - optimizer mode and applied overlays;
   - proposal v1 versus final proposal when rerun;
   - risk-controller and reviewer decisions.

7. **Appendix**
   - full holding reports;
   - raw analytics and market-context JSON;
   - detailed agent reviews and diagnostics.

### Persistence requirements

Persist both human-readable and machine-readable results:

```text
reports/portfolio_<timestamp>/
  complete_report.md
  final_result.json
  holdings/
  portfolio/
    market_regime.json
    data_quality.json
    analytics.json
    proposals/
      proposal_v1.json
      proposal_v2.json
    risk_validation.json
    proposal_review.json
    approval.json
```

---

## 6. Agent Responsibility Matrix

| Agent / Component | Primary responsibility | May produce target weights? | Required output |
| --- | --- | ---: | --- |
| Instrument analysts and debates | Analyze one instrument | No | Evidence and structured instrument proposal |
| Instrument Decision Manager | Produce initial instrument rating and conviction | No | `InstrumentProposal` |
| Market Regime Analyst | Interpret benchmark trend and recent news | No | `MarketRegime` |
| Deterministic Analytics Engine | Compute portfolio facts | No | `PortfolioAnalytics` |
| Deterministic Allocation Engine | Generate and validate target weights | **Yes** | `RebalanceProposal` |
| Portfolio Risk Controller | Validate whole-portfolio risk and recommend constraints | No | `RiskValidationResult` |
| Allocation Proposal Reviewer | Approve, reject, or request deterministic rerun | No | `ProposalReview` |
| Portfolio Decision Approver | Approve immutable final proposal and explain it | No | `PortfolioApprovalDecision` |
| Final Result Validator | Enforce output invariants | No | `FinalRebalanceResult` |

Recommended naming changes should be introduced with compatibility aliases where needed:

- single-instrument `Portfolio Manager` → `Instrument Decision Manager`;
- portfolio `Risk Analyst` → `Portfolio Risk Controller`;
- portfolio `Rebalancer` → `Allocation Proposal Reviewer`;
- portfolio `Allocation Manager` → `Portfolio Decision Approver`.

---

## 7. State and Schema Plan

Introduce a typed workflow state rather than passing an open-ended dictionary:

```python
class PortfolioWorkflowState(TypedDict):
    run_id: str
    request: PortfolioRequest
    effective_constraints: EffectivePortfolioConstraints
    holding_analyses: list[HoldingAnalysisResult]
    instrument_proposals: dict[str, InstrumentProposal]
    analytics_inputs: PortfolioAnalyticsInputs
    market_regime: MarketRegime
    data_quality: DataQualityAssessment
    analytics: PortfolioAnalytics
    proposals: list[RebalanceProposal]
    risk_validation: RiskValidationResult
    proposal_review: ProposalReview
    approval: PortfolioApprovalDecision
    final_result: FinalRebalanceResult
    warnings: list[str]
    events: list[PortfolioWorkflowEvent]
```

Each node must declare and test its required inputs and outputs. Markdown should be generated only by renderers, not used as the primary inter-agent data contract.

---

## 8. Implementation Phases

## Phase A: Extract and Clarify the Existing Workflow

### Goal

Separate business orchestration from CLI interaction without changing allocation behavior.

### Tasks

- add `tradingagents/portfolio/workflow.py` or `tradingagents/graph/portfolio_graph.py`;
- move the current four-stage portfolio run out of `cli/main.py`;
- expose a reusable `run_portfolio_workflow(...)` entry point;
- introduce typed workflow state and progress events;
- keep CLI responsible only for collecting input, rendering progress, saving, and displaying;
- clarify agent names in CLI and reports.

### Acceptance criteria

- CLI and programmatic callers use the same workflow entry point;
- existing portfolio tests remain green;
- no portfolio-level agent is directly constructed by the CLI;
- current output behavior remains backward compatible.

---

## Phase B: Structured Instrument Proposals and Data Quality

### Goal

Replace rating-from-Markdown as the primary portfolio input and make missing-data behavior explicit.

### Tasks

- add `InstrumentProposal` and structured instrument output;
- retain text parsing as a fallback;
- add conviction, horizon, risk flags, and confidence;
- add `DataQualityAssessment` and configurable policies;
- restrict increases for failed or unsupported holdings;
- add conservative option handling when contract-level evidence is unavailable.

### Acceptance criteria

- the optimizer consumes structured proposals when available;
- failed holding analysis cannot silently become a positive allocation signal;
- final reports display weighted analysis and metric coverage;
- restricted and insufficient-data paths have focused tests.

---

## Phase C: Structured Market Regime and Deterministic Market Overlay

### Goal

Make broad-market trend and recent news affect target weights through validated deterministic inputs.

### Tasks

- add `MarketRegime` schema and market-regime agent;
- derive deterministic benchmark trend features where possible;
- convert market interpretation into bounded risk, cash, and sector overlays;
- validate overlays before use;
- pass validated overlays into the deterministic allocation engine;
- display the market-regime summary and allocation implications in reports.

### Acceptance criteria

- risk-off and risk-on scenarios produce predictably different deterministic proposals;
- market overlay cannot violate user hard constraints;
- low-confidence market context has reduced or no allocation impact;
- market/news influence is visible in proposal diagnostics.

---

## Phase D: Proposal Integrity and Agent Decision Protocol

### Goal

Ensure agents can challenge allocation without inventing final weights.

### Tasks

- add immutable proposal IDs, versions, and input hashes;
- add structured `RiskValidationResult`, `ProposalReview`, and `PortfolioApprovalDecision`;
- remove target-weight generation from the final approval agent;
- add final-result validation;
- preserve a compatibility Markdown renderer for existing reports and callers.

### Acceptance criteria

- every approved decision references a deterministic proposal;
- final displayed weights exactly equal approved proposal weights;
- all symbol, weight-sum, action, and constraint invariants are validated;
- malformed LLM output cannot produce an executable trade list.

---

## Phase E: Bounded Risk Review and Re-optimization Loop

### Goal

Allow portfolio-level review to materially improve the allocation while keeping all numerical changes deterministic.

### Tasks

- allow Risk Controller and Reviewer to request bounded constraint adjustments;
- validate requested adjustments;
- rerun optimizer with proposal versioning;
- rerun final risk validation;
- cap reruns, initially at one;
- preserve proposal history and reasons.

### Acceptance criteria

- a risk failure can trigger a second deterministic proposal;
- invalid agent-requested constraints are rejected;
- no unbounded workflow loop is possible;
- proposal history clearly explains why final weights changed.

---

## Phase F: Decision-First Reporting and User Experience

### Goal

Make the final result easy to understand and audit.

### Tasks

- redesign CLI output around executive decision and final trade list;
- add explicit market-regime, data-quality, and portfolio-risk summaries;
- add one row per instrument showing initial proposal versus final action;
- move verbose agent narratives to appendices;
- write machine-readable final result and proposal-history files;
- update README and example output.

### Acceptance criteria

A user can answer the following from the first report sections alone:

- should the portfolio rebalance now?
- what should be bought, added, held, trimmed, or sold?
- what are the exact current and target weights?
- how did market trend, news, portfolio risk, and initial instrument views affect the result?
- what limitations or missing data reduce confidence?

---

## 9. Testing Strategy

## 9.1 Unit tests

Add focused tests for:

- structured instrument proposal validation;
- rating, conviction, and confidence normalization;
- market-regime schema and overlay bounds;
- data-quality policy decisions;
- risk-controller and reviewer structured outputs;
- proposal immutability and versioning;
- final-result validation;
- report rendering.

## 9.2 Deterministic allocation tests

Required scenarios:

1. same instrument views under `risk_on` versus `risk_off` market regimes;
2. negative sector overlay reducing, but not arbitrarily eliminating, sector allocation;
3. high correlation and concentration reducing otherwise positive proposals;
4. user hard constraints overriding agent overlays;
5. failed analysis preventing an increase;
6. missing option contract data preventing an option increase;
7. low-confidence market regime having limited allocation impact;
8. target weights always summing to 100%.

## 9.3 Workflow tests

Required scenarios:

- complete happy-path portfolio workflow;
- partial provider failure with restricted rebalance;
- insufficient-data Hold-only result;
- risk-controller failure causing proposal v2;
- reviewer rejection without executable trades;
- malformed agent output rejected by schema validation;
- CLI and programmatic entry points returning the same final result.

## 9.4 Report tests

Validate that the complete report includes:

- market trend summary;
- final rebalance table;
- initial instrument ratings and summaries;
- portfolio-risk summary;
- rebalance rationale;
- data-quality warnings;
- proposal version history when applicable.

---

## 10. Migration and Backward Compatibility

- preserve existing `propagate()` single-instrument behavior;
- preserve `propagate_portfolio()` during migration, but make it an internal stage of the new workflow;
- preserve existing Markdown fields while introducing structured objects alongside them;
- provide compatibility aliases for renamed agents and state keys;
- keep existing report files during the transition and add new machine-readable outputs;
- introduce feature flags for market overlays and re-optimization until the new workflow is stable;
- document any change in default optimizer behavior before enabling it by default.

---

## 11. Observability and Audit Requirements

Every portfolio run should record:

- run ID, trade date, model/provider configuration, and effective constraints;
- data-source warnings and weighted coverage;
- instrument proposal inputs and confidence;
- market-regime input, confidence, and overlays;
- proposal input hash, optimizer mode, diagnostics, and constraint applications;
- risk-controller decision;
- reviewer decision and requested changes;
- proposal versions and rejection/rerun reasons;
- final approval and validation status.

Progress events should use stable stage names rather than CLI-only `Stage 1/4` text so UIs, APIs, and tests can consume them consistently.

---

## 12. Recommended Initial Delivery Slice

The first implementation slice should improve clarity and correctness without changing the optimizer's strategy:

1. extract the current portfolio workflow from the CLI into a reusable orchestrator;
2. introduce `PortfolioWorkflowState` and typed result objects;
3. rename the portfolio-level reviewer and approver roles in user-facing output;
4. make the final approver approve or reject an immutable deterministic proposal rather than regenerate weights;
5. add final-result validation and a decision-first report;
6. add end-to-end workflow contract tests.

After this foundation is stable, implement structured market regime, deterministic market overlays, data-quality restrictions, and bounded re-optimization.

---

## 13. Definition of Done

The optimized workflow is complete when:

- every instrument produces a structured initial proposal or explicit failure state;
- market trend and recent news produce a structured, confidence-scored market regime;
- portfolio analytics include the available exposure and risk dimensions;
- market regime, instrument proposals, portfolio risk, current weights, and user constraints all influence final weights through deterministic code;
- portfolio agents can approve, reject, or request a bounded deterministic rerun, but cannot directly invent target weights;
- one validated immutable proposal is the authoritative final rebalance result;
- the final report clearly contains instrument summaries, current market trend, portfolio risk, rebalance rationale, exact target weights, actions, constraints, and warnings;
- focused unit, workflow, and report tests prove the required invariants and failure policies.
