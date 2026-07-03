# QuantV2 Codex Instructions

QuantV2 is an event-driven equity quant research system.

The goal is not live trading yet. The goal is to build a clean, reproducible research pipeline that prevents lookahead bias.

## Core Principle

Every feature must be known at or before the decision timestamp.

Every label must be generated strictly after the decision timestamp.

Never use future data, revised data, final outcome data, or post-event knowledge when building features.

## Current MVP

MVP 1 is complete enough to support an end-to-end event-study research loop.

Current phase: MVP 2 — baseline prediction and honest walk-forward evaluation.

MVP 2 goals:

1. Walk-forward time-series splitting
2. Rule-based baseline prediction
3. Baseline prediction evaluation
4. No-trade handling
5. Simple cost-aware reporting

Do not build live trading, brokerage integration, autonomous execution, complex AI reasoning, or historical analog models yet.

## Quant Rules

- Always distinguish prediction time from outcome time.
- Always preserve point-in-time correctness.
- Every feature must be available before the label window begins.
- Every label must be generated after the decision timestamp.
- Do not optimize for an impressive backtest.
- Optimize for honest out-of-sample evaluation.
- Do not rely only on directional accuracy.
- Include transaction costs before claiming a strategy works.
- Treat no-trade decisions as first-class model outputs.

## Python Rules

- Use Python 3.
- Use pandas and numpy for data work.
- Use scikit-learn only for simple baseline models at first.
- Use type hints where practical.
- Prefer explicit function arguments over hidden global state.
- Keep functions deterministic when possible.
- Add docstrings to public functions.
- Avoid unnecessary dependencies.

## Architecture Rules

Keep modules small and focused.

Use this structure:

- src/quantv2/data/ for data loading
- src/quantv2/labels/ for label generation
- src/quantv2/features/ for feature creation
- src/quantv2/models/ for baseline models
- src/quantv2/backtest/ for walk-forward testing and simulation
- src/quantv2/evaluation/ for metrics and reports
- tests/ for automated tests
- configs/ for strategy and data settings

Do not mix research notebooks with reusable source code.

## Testing Rules

- Add tests when changing label logic, feature logic, or backtest logic.
- Add no-lookahead tests when creating labels or features.
- Run pytest after meaningful code changes.
- Never claim tests passed unless they actually ran.
- If tests fail, explain the failure and make the smallest reasonable fix.

## Data Rules

- Do not commit raw market data.
- Do not commit paid data.
- Do not commit API keys.
- Do not commit brokerage credentials.
- Do not commit .env files.
- Test data must be tiny, synthetic, and safe to commit.

## Git Rules

- Make small focused commits.
- Explain what changed and why.
- Before large edits, provide a short plan.
- Do not rewrite project structure without asking first.

## Preferred Workflow

For each task:

1. Inspect the relevant files.
2. Make a short plan.
3. Implement the smallest useful change.
4. Add or update tests.
5. Run the relevant tests.
6. Summarize files changed, tests run, and remaining risks.

## Current Priority

The next real implementation target is:

src/quantv2/evaluation/prediction_metrics.py

This module should evaluate deterministic baseline predictions against existing forward-return label columns.

The prediction evaluation layer must ensure:

- It only evaluates existing prediction columns and existing forward_return_Nd label columns.
- It never creates new labels from prices or future data.
- It never uses label_date columns as features.
- It never trains a model.
- It never creates live trades.
- It never connects to a brokerage.
- It treats no-trade decisions as first-class outputs.
- It reports directional hit rate, coverage, trade count, no-trade count, and signed forward-return summaries.
- It must not claim profitability, because transaction costs and execution simulation are not implemented yet.
