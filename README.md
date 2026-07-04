# QuantV2

QuantV2 is an event-driven equity quant research system. Its current purpose is
to build a clean, reproducible command-line research pipeline for daily equity
event studies and honest walk-forward baseline evaluation.

This is not live trading software. QuantV2 does not place orders, connect to a
brokerage account, simulate real execution, or claim strategy profitability. The
saved outputs are research artifacts only. Cost-adjusted returns are simple
research estimates based on user-supplied basis-point assumptions, not real PnL.

## Current Workflow

The current MVP pipeline runs from local CSV files:

1. Load a required market CSV.
2. Load an optional event CSV.
3. Validate required columns, dates, duplicates, OHLC relationships, and missing
   values.
4. Build backward-looking price features.
5. Build close-to-close forward-return labels.
6. Attach event features by exact `ticker` and `decision_date` when event data is
   provided.
7. Create an event-study report.
8. Create walk-forward train/test splits using decision dates.
9. Generate deterministic rule baseline predictions.
10. Evaluate predictions, including no-trade outputs.
11. Apply simple transaction-cost assumptions.
12. Save experiment CSV artifacts and a JSON manifest.

## Point-In-Time Assumptions

Every feature must be known at or before `decision_date`. Every label must occur
strictly after `decision_date`.

The current daily feature builders treat each market row as a decision row. They
assume the decision is made after the market close on `decision_date`, so fields
derived from that day's open, close, and prior rows are considered available.
Forward-return labels use future rows for the same ticker, and each
`label_date_Nd` must be later than `decision_date`.

Events are point-in-time inputs supplied by the user. If an event CSV is used,
each event row must represent information that was known by its
`decision_date`.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
python -m pytest
```

`python -m pip install -e .` installs the package in editable mode and installs
the runtime dependencies declared in `pyproject.toml`. `pytest` is installed
separately for the test suite.

## Input CSVs

Market data is required. The market CSV must include:

- `ticker`
- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

Market rows are validated for non-empty tickers, valid dates, no duplicate
`ticker` plus `date` rows, numeric OHLCV fields, positive OHLC values,
non-negative volume, and internally consistent OHLC relationships.

Event data is optional. When provided, the event CSV must include:

- `ticker`
- `decision_date`
- `event_type`
- `event_direction`
- `event_severity`

Event rows are validated for non-empty required fields, valid decision dates,
and no duplicate `ticker` plus `decision_date` rows. Events are joined to the
research matrix by exact `ticker` and `decision_date`.

## Create Sample Inputs

The ignored `data/raw/` directory is a convenient place for local sample inputs.
This example creates tiny synthetic CSVs that are safe to keep local.

```bash
mkdir -p data/raw

python - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("data/raw")
dates = pd.date_range("2024-01-01", periods=18, freq="D")
rows = []

for ticker, base_close, close_multiplier, gap_rate in (
    ("AAA", 100.0, 1.03, 0.02),
    ("BBB", 200.0, 0.98, -0.02),
    ("CCC", 50.0, 1.00, 0.00),
):
    previous_close = None
    for index, date in enumerate(dates):
        close = base_close * (close_multiplier ** index)
        open_price = close if previous_close is None else previous_close * (1 + gap_rate)
        high = max(open_price, close) * 1.01
        low = min(open_price, close) * 0.99
        rows.append({
            "ticker": ticker,
            "date": date.strftime("%Y-%m-%d"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000 + index,
        })
        previous_close = close

pd.DataFrame(rows).to_csv(out / "sample_market_data.csv", index=False)

pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC"],
    "decision_date": ["2024-01-10", "2024-01-11", "2024-01-13"],
    "event_type": ["earnings", "guidance", "filing"],
    "event_direction": ["positive", "negative", "neutral"],
    "event_severity": [3.0, 2.0, 1.0],
}).to_csv(out / "sample_event_data.csv", index=False)
PY
```

## CLI Usage

Show the available command-line options:

```bash
python -m quantv2.experiments.cli --help
```

Run a complete saved walk-forward baseline experiment with market and event
data:

```bash
python -m quantv2.experiments.cli \
  --market-data data/raw/sample_market_data.csv \
  --event-data data/raw/sample_event_data.csv \
  --output-dir data/experiments \
  --experiment-name local_smoke_test \
  --run-id first_run \
  --train-window 8 \
  --test-window 4 \
  --step-size 4 \
  --return-windows 5,20 \
  --volatility-window 20 \
  --commission-bps 0.0 \
  --slippage-bps 5.0 \
  --extra-cost-bps 0.0
```

For larger daily datasets, the defaults use horizons of `1,3,5`, return windows
of `5,20`, a `252` decision-date training window, and a `21` decision-date test
window. The tiny sample above uses shorter split windows so at least one
walk-forward split can be created.

To run without event data, omit `--event-data`. The pipeline will still create
price features, forward-return labels, market-only event-study groupings,
walk-forward splits, baseline predictions, and evaluation artifacts.

## Saved Artifacts

Runs are saved under:

```text
data/experiments/<experiment-name>/<run-id>/
```

Each run directory contains:

- `manifest.json`: run metadata, artifact names, row counts, and saved columns.
- `market_data.csv`: validated market data.
- `event_data.csv`: validated event data, only when `--event-data` is provided.
- `feature_matrix.csv`: point-in-time price features plus forward-return labels.
- `research_data.csv`: feature matrix with event fields attached when available.
- `event_study_report.csv`: grouped event-study summary statistics.
- `splits.csv`: walk-forward train/test split metadata.
- `predictions.csv`: rule baseline outputs, no-trade reasons, split metadata,
  signed returns, and simple cost-adjusted return estimates.
- `prediction_summary.csv`: prediction metrics by split and horizon.
- `cost_adjusted_summary.csv`: simple cost-adjusted return summaries by split
  and horizon.

`data/raw/` and `data/experiments/` are ignored by Git, except for their
`.gitkeep` placeholders. Do not commit raw market data, paid data, API keys,
brokerage credentials, `.env` files, or generated experiment outputs.

## Current Limitations

- No live trading.
- No brokerage integration.
- No real execution simulator.
- No PnL or profitability claims.
- Simple deterministic rule baseline only.
- Simple transaction-cost assumptions only.
- No learned model yet.

## Next Roadmap

- Improve event schemas.
- Add stricter data quality checks.
- Add baseline model comparison.
- Add calibration.
- Add better cost and risk modeling.
