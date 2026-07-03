# QuantV2

QuantV2 is an event-driven equity quant research system.

The first goal is not live trading. The first goal is to build a clean research pipeline that prevents lookahead bias.

## Core Principle

Every feature must be known at or before the decision timestamp.

Every label must be generated strictly after the decision timestamp.

## MVP 1

The first MVP is Data + Label Integrity.

Goals:

- Load market data
- Load event data
- Generate forward-return labels
- Build a feature matrix
- Add no-lookahead tests
- Produce a basic event-study report

## Initial Strategy Focus

Event-driven daily equities.

Initial holding periods:

- 1 trading day
- 3 trading days
- 5 trading days

## Repo Structure

- `configs/` strategy and data settings
- `data/` local data storage, ignored by Git
- `src/quantv2/` source code
- `tests/` automated tests
- `notebooks/` research notebooks
