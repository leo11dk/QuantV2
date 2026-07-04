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

src/quantv2/experiments/registry.py

This module should save experiment outputs and metadata for reproducible research.

The experiment registry layer must ensure:

- It saves DataFrame outputs from experiment runners to local files.
- It writes a manifest file describing the run.
- It records experiment name, run ID, created timestamp, metadata, artifact names, row counts, and columns.
- It never fetches live data.
- It never creates labels from prices or future data.
- It never trains models.
- It never connects to a brokerage.
- It never creates live trades.
- It never creates orders, executions, fills, positions, PnL, or profit claims.
- It must not mutate input DataFrames.
- It should write only to local experiment output directories such as data/experiments.