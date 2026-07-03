from __future__ import annotations

import pandas as pd

from quantv2.features.price_features import add_price_features
from quantv2.labels.forward_returns import add_forward_return_labels


def build_feature_matrix(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    return_windows: tuple[int, ...] = (5, 20),
    volatility_window: int = 20,
    ticker_col: str = "ticker",
    date_col: str = "date",
    open_col: str = "open",
    close_col: str = "close",
) -> pd.DataFrame:
    """Build a research matrix from point-in-time features and future labels."""
    featured = add_price_features(
        prices=prices,
        return_windows=return_windows,
        volatility_window=volatility_window,
        ticker_col=ticker_col,
        date_col=date_col,
        open_col=open_col,
        close_col=close_col,
    )
    labeled = add_forward_return_labels(
        prices=prices,
        horizons=horizons,
        ticker_col=ticker_col,
        date_col=date_col,
        price_col=close_col,
    )

    alignment_columns = [ticker_col, date_col, "decision_date"]
    if not featured[alignment_columns].equals(labeled[alignment_columns]):
        raise ValueError("feature and label rows are not aligned by ticker and decision_date")

    feature_columns = [
        "prev_close",
        "gap_pct",
        *(f"prior_{window}d_return" for window in return_windows),
        f"volatility_{volatility_window}d",
    ]
    label_columns = [
        column
        for horizon in horizons
        for column in (
            f"label_date_{horizon}d",
            f"close_{horizon}d",
            f"forward_return_{horizon}d",
        )
    ]

    matrix = pd.concat(
        [
            featured[[ticker_col, "decision_date", *feature_columns]],
            labeled[label_columns],
        ],
        axis=1,
    )

    for horizon in horizons:
        label_date_col = f"label_date_{horizon}d"
        has_label_date = matrix[label_date_col].notna()
        if not (
            matrix.loc[has_label_date, label_date_col]
            > matrix.loc[has_label_date, "decision_date"]
        ).all():
            raise ValueError(
                f"{label_date_col} must be strictly after decision_date when available"
            )

    return matrix.copy(deep=True)
