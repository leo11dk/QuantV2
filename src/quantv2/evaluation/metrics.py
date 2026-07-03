from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = [
    "horizon",
    "count",
    "missing_count",
    "mean_return",
    "median_return",
    "std_return",
    "min_return",
    "max_return",
    "hit_rate",
]


def summarize_forward_returns(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    group_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Summarize already-created forward-return label columns.

    This function does not create labels, inspect price columns, or infer missing
    returns. It only aggregates existing ``forward_return_Nd`` columns.
    """
    label_columns = [f"forward_return_{horizon}d" for horizon in horizons]
    missing_label_columns = [
        column for column in label_columns if column not in data.columns
    ]
    if missing_label_columns:
        missing = ", ".join(missing_label_columns)
        raise ValueError(f"data is missing required forward-return columns: {missing}")

    if group_cols is not None:
        missing_group_columns = [
            column for column in group_cols if column not in data.columns
        ]
        if missing_group_columns:
            missing = ", ".join(missing_group_columns)
            raise ValueError(f"data is missing group columns: {missing}")

    rows: list[dict[str, object]] = []

    for horizon, label_column in zip(horizons, label_columns):
        if group_cols is None:
            rows.append(_summary_row(data[label_column], horizon))
            continue

        groupby_arg: str | list[str]
        groupby_arg = group_cols[0] if len(group_cols) == 1 else list(group_cols)
        grouped = data.groupby(groupby_arg, sort=False, dropna=False)

        for group_key, group in grouped:
            key_values = (
                (group_key,)
                if len(group_cols) == 1
                else tuple(group_key)
            )
            row = {
                group_col: key_value
                for group_col, key_value in zip(group_cols, key_values)
            }
            row.update(_summary_row(group[label_column], horizon))
            rows.append(row)

    columns = SUMMARY_COLUMNS if group_cols is None else [*group_cols, *SUMMARY_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def _summary_row(returns: pd.Series, horizon: int) -> dict[str, object]:
    valid_returns = returns.dropna()

    return {
        "horizon": horizon,
        "count": int(valid_returns.count()),
        "missing_count": int(returns.isna().sum()),
        "mean_return": valid_returns.mean(),
        "median_return": valid_returns.median(),
        "std_return": valid_returns.std(),
        "min_return": valid_returns.min(),
        "max_return": valid_returns.max(),
        "hit_rate": valid_returns.gt(0.0).mean(),
    }
