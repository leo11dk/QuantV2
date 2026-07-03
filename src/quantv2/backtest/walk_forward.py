from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """Train/test index selections for one point-in-time walk-forward split."""

    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: pd.Index
    test_indices: pd.Index


def make_walk_forward_splits(
    data: pd.DataFrame,
    date_col: str = "decision_date",
    train_window: int = 252,
    test_window: int = 21,
    step_size: int | None = None,
    min_train_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create rolling walk-forward train/test splits using decision dates only.

    Window sizes are measured in unique decision dates rather than rows. The
    returned indices select rows from the original input without mutating it.
    """

    if train_window <= 0:
        raise ValueError("train_window must be greater than 0")
    if test_window <= 0:
        raise ValueError("test_window must be greater than 0")

    resolved_step_size = test_window if step_size is None else step_size
    if resolved_step_size <= 0:
        raise ValueError("step_size must be greater than 0")

    resolved_min_train_size = train_window if min_train_size is None else min_train_size
    if resolved_min_train_size <= 0:
        raise ValueError("min_train_size must be greater than 0")

    if date_col not in data.columns:
        raise ValueError(f"missing required column: {date_col}")

    decision_dates = pd.to_datetime(data[date_col])
    unique_dates = pd.Index(decision_dates.dropna().unique()).sort_values()

    splits: list[WalkForwardSplit] = []
    test_start_position = resolved_min_train_size

    while test_start_position + test_window <= len(unique_dates):
        train_start_position = max(0, test_start_position - train_window)
        train_dates = unique_dates[train_start_position:test_start_position]
        test_dates = unique_dates[test_start_position : test_start_position + test_window]

        if len(train_dates) >= resolved_min_train_size:
            train_mask = decision_dates.isin(train_dates)
            test_mask = decision_dates.isin(test_dates)
            train_indices = pd.Index(data.index[train_mask.to_numpy()])
            test_indices = pd.Index(data.index[test_mask.to_numpy()])

            if train_indices.intersection(test_indices).size > 0:
                raise ValueError("train_indices and test_indices must not overlap")
            if train_dates[-1] >= test_dates[0]:
                raise ValueError("train dates must be strictly before test dates")

            splits.append(
                WalkForwardSplit(
                    split_id=len(splits),
                    train_start=pd.Timestamp(train_dates[0]),
                    train_end=pd.Timestamp(train_dates[-1]),
                    test_start=pd.Timestamp(test_dates[0]),
                    test_end=pd.Timestamp(test_dates[-1]),
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )

        test_start_position += resolved_step_size

    if not splits:
        raise ValueError("not enough unique dates to create at least one split")

    return splits
