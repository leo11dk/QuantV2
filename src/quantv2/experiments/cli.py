from __future__ import annotations

import argparse
from pathlib import Path

from quantv2.experiments.run_experiment import (
    run_and_save_walk_forward_baseline_experiment,
)


def _parse_comma_separated_ints(option_name: str):
    def parse(value: str) -> tuple[int, ...]:
        parts = value.split(",")
        parsed_values: list[int] = []
        for part in parts:
            stripped = part.strip()
            if stripped == "":
                raise argparse.ArgumentTypeError(
                    f"{option_name} must be a comma-separated list of integers"
                )
            try:
                parsed_values.append(int(stripped))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"{option_name} must be a comma-separated list of integers"
                ) from exc

        if not parsed_values:
            raise argparse.ArgumentTypeError(
                f"{option_name} must be a comma-separated list of integers"
            )
        return tuple(parsed_values)

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and save the walk-forward baseline research experiment.",
    )
    parser.add_argument("--market-data", required=True, type=Path)
    parser.add_argument("--event-data", type=Path)
    parser.add_argument("--output-dir", default=Path("data/experiments"), type=Path)
    parser.add_argument("--experiment-name", default="walk_forward_baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--horizons",
        default=(1, 3, 5),
        type=_parse_comma_separated_ints("--horizons"),
    )
    parser.add_argument(
        "--return-windows",
        default=(5, 20),
        type=_parse_comma_separated_ints("--return-windows"),
    )
    parser.add_argument("--volatility-window", default=20, type=int)
    parser.add_argument("--train-window", default=252, type=int)
    parser.add_argument("--test-window", default=21, type=int)
    parser.add_argument("--step-size", type=int)
    parser.add_argument("--min-train-size", type=int)
    parser.add_argument("--gap-threshold", default=0.01, type=float)
    parser.add_argument("--momentum-threshold", default=0.02, type=float)
    parser.add_argument("--min-score-to-trade", default=2, type=int)
    parser.add_argument("--max-volatility", type=float)
    parser.add_argument("--commission-bps", default=0.0, type=float)
    parser.add_argument("--slippage-bps", default=5.0, type=float)
    parser.add_argument("--extra-cost-bps", default=0.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the saved walk-forward baseline experiment from local CSV inputs."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    prediction_kwargs = {
        "gap_threshold": args.gap_threshold,
        "momentum_threshold": args.momentum_threshold,
        "min_score_to_trade": args.min_score_to_trade,
    }
    if args.max_volatility is not None:
        prediction_kwargs["max_volatility"] = args.max_volatility

    cost_kwargs = {
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "extra_cost_bps": args.extra_cost_bps,
    }

    results, run_dir = run_and_save_walk_forward_baseline_experiment(
        market_data_path=args.market_data,
        event_data_path=args.event_data,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        run_id=args.run_id,
        overwrite=args.overwrite,
        horizons=args.horizons,
        return_windows=args.return_windows,
        volatility_window=args.volatility_window,
        train_window=args.train_window,
        test_window=args.test_window,
        step_size=args.step_size,
        min_train_size=args.min_train_size,
        prediction_kwargs=prediction_kwargs,
        cost_kwargs=cost_kwargs,
    )

    print(f"Saved run directory: {run_dir}")
    print("Artifacts:")
    for artifact_name in results:
        print(f"- {artifact_name}.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
