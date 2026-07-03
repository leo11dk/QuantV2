# QuantV2 Codex Instructions

QuantV2 is an event-driven equity quant research system.

The goal is not live trading yet. The goal is to build a clean, reproducible research pipeline that prevents lookahead bias.

## Core Principle

Every feature must be known at or before the decision timestamp.

Every label must be generated strictly after the decision timestamp.

Never use future data, revised data, final outcome data, or post-event knowledge when building features.

## Current MVP

Build MVP 1 only:

1. Repo scaffold
2. Data schemas
3. Forward-return labels
4. Feature matrix
5. No-lookahead tests
6. Basic event-study evaluation

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

src/quantv2/data/event_data.py

This module should load and validate structured event data for the QuantV2 research pipeline.

The event data layer must ensure:

- Required columns are present: ticker, decision_date, event_type, event_direction, event_severity.
- Dates are parsed consistently.
- Rows are sorted by ticker and decision_date.
- Duplicate ticker/decision_date rows are rejected for now.
- Missing ticker or decision_date values are rejected.
- Missing required event fields are rejected.
- Extra metadata columns such as source, source_reliability, notes, or event_time are preserved.
- The loader must not create labels.
- The loader must not create price features.
- The loader must not create trading signals.
- The loader must not fetch live data yet.
- The output should be clean enough to pass into src/quantv2/features/event_features.py.
