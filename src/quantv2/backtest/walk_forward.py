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


@dataclass(frozen=True)
class PurgedWalkForwardSplit:
    """Walk-forward split with training rows purged by label availability."""

    split_id: int
    horizon: int
    label_date_col: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: pd.Index
    test_indices: pd.Index
    candidate_train_row_count: int
    usable_train_row_count: int
    purged_train_row_count: int


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


def make_purged_walk_forward_splits(
    data: pd.DataFrame,
    horizon: int,
    date_col: str = "decision_date",
    label_date_col: str | None = None,
    train_window: int = 252,
    test_window: int = 21,
    step_size: int | None = None,
    min_train_size: int | None = None,
    min_usable_train_rows: int = 1,
) -> list[PurgedWalkForwardSplit]:
    """Create label-availability-aware purged walk-forward splits.

    Candidate windows come from :func:`make_walk_forward_splits`. A candidate
    training row is retained only when its label date is known strictly before
    the corresponding test window starts. Window sizes are measured in unique
    decision dates by the ordinary splitter.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if (
        isinstance(min_usable_train_rows, bool)
        or not isinstance(min_usable_train_rows, int)
        or min_usable_train_rows <= 0
    ):
        raise ValueError("min_usable_train_rows must be a positive integer")

    resolved_label_date_col = (
        f"label_date_{horizon}d" if label_date_col is None else label_date_col
    )

    if date_col not in data.columns:
        raise ValueError(f"missing required column: {date_col}")
    if resolved_label_date_col not in data.columns:
        raise ValueError(f"missing required column: {resolved_label_date_col}")

    decision_dates = pd.to_datetime(data[date_col])
    label_dates = pd.to_datetime(data[resolved_label_date_col])

    non_missing_label_dates = label_dates.notna()
    invalid_label_dates = non_missing_label_dates & ~label_dates.gt(decision_dates)
    if invalid_label_dates.any():
        raise ValueError(
            f"non-missing {resolved_label_date_col} values must be strictly "
            f"after {date_col}"
        )

    candidate_splits = make_walk_forward_splits(
        data=data,
        date_col=date_col,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
        min_train_size=min_train_size,
    )

    purged_splits: list[PurgedWalkForwardSplit] = []

    for candidate_split in candidate_splits:
        candidate_train_mask = data.index.isin(candidate_split.train_indices)
        usable_train_mask = (
            candidate_train_mask
            & label_dates.notna().to_numpy()
            & label_dates.lt(candidate_split.test_start).to_numpy()
        )
        usable_train_indices = pd.Index(data.index[usable_train_mask])

        candidate_train_row_count = len(candidate_split.train_indices)
        usable_train_row_count = len(usable_train_indices)
        purged_train_row_count = candidate_train_row_count - usable_train_row_count

        if usable_train_row_count < min_usable_train_rows:
            continue

        purged_splits.append(
            PurgedWalkForwardSplit(
                split_id=candidate_split.split_id,
                horizon=horizon,
                label_date_col=resolved_label_date_col,
                train_start=candidate_split.train_start,
                train_end=candidate_split.train_end,
                test_start=candidate_split.test_start,
                test_end=candidate_split.test_end,
                train_indices=usable_train_indices,
                test_indices=candidate_split.test_indices,
                candidate_train_row_count=candidate_train_row_count,
                usable_train_row_count=usable_train_row_count,
                purged_train_row_count=purged_train_row_count,
            )
        )

    if not purged_splits:
        raise ValueError("no usable purged walk-forward splits remain")

    return purged_splits
