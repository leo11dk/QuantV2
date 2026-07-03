from __future__ import annotations

import pandas as pd

from quantv2.evaluation.metrics import SUMMARY_COLUMNS, summarize_forward_returns


REPORT_METADATA_COLUMNS = ["report_section", "grouping"]


def build_event_study_report(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 3, 5),
    groupings: tuple[tuple[str, ...], ...] | None = None,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Build an event-study report from existing forward-return labels.

    The report delegates all metric calculations to ``summarize_forward_returns``.
    It does not create labels, infer missing returns, or create trading signals.
    """
    requested_groupings = groupings or ()
    if not include_overall and not requested_groupings:
        raise ValueError("include_overall=False requires at least one grouping")

    _validate_groupings(data, requested_groupings)

    sections: list[pd.DataFrame] = []
    grouping_columns = _grouping_columns(requested_groupings)

    if include_overall:
        overall = summarize_forward_returns(data, horizons=horizons)
        overall.insert(0, "grouping", "overall")
        overall.insert(0, "report_section", "overall")
        sections.append(overall)

    for grouping in requested_groupings:
        grouped = summarize_forward_returns(
            data,
            horizons=horizons,
            group_cols=grouping,
        )
        grouped.insert(0, "grouping", _grouping_name(grouping))
        grouped.insert(0, "report_section", _section_name(grouping))
        sections.append(grouped)

    report = pd.concat(sections, ignore_index=True, sort=False)
    columns = [*REPORT_METADATA_COLUMNS, *grouping_columns, *SUMMARY_COLUMNS]
    report = report.reindex(columns=columns)

    sort_columns = [
        column
        for column in ("report_section", *grouping_columns, "horizon")
        if column in report.columns
    ]
    return report.sort_values(
        sort_columns,
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)


def _validate_groupings(
    data: pd.DataFrame,
    groupings: tuple[tuple[str, ...], ...],
) -> None:
    missing_columns: list[str] = []

    for grouping in groupings:
        if not grouping:
            raise ValueError("groupings must contain non-empty column tuples")

        for column in grouping:
            if column not in data.columns and column not in missing_columns:
                missing_columns.append(column)

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"data is missing grouping columns: {missing}")


def _grouping_columns(groupings: tuple[tuple[str, ...], ...]) -> list[str]:
    columns: list[str] = []

    for grouping in groupings:
        for column in grouping:
            if column not in columns:
                columns.append(column)

    return columns


def _section_name(grouping: tuple[str, ...]) -> str:
    return f"by_{'_and_'.join(grouping)}"


def _grouping_name(grouping: tuple[str, ...]) -> str:
    return ",".join(grouping)
