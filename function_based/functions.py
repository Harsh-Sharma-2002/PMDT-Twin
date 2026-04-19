from __future__ import annotations

from typing import Any

import pandas as pd


CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
AMOUNT_COL = "case:Amount"


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _case_duration_hours(case_df: pd.DataFrame) -> float:
    if TIME_COL not in case_df.columns:
        return 0.0

    ts = _to_datetime(case_df[TIME_COL]).dropna()
    if len(ts) < 2:
        return 0.0

    return (ts.max() - ts.min()).total_seconds() / 3600.0


def _median_case_duration_hours(df: pd.DataFrame) -> float:
    if CASE_COL not in df.columns or TIME_COL not in df.columns:
        return 0.0

    durations: list[float] = []
    for _, group in df.groupby(CASE_COL):
        d = _case_duration_hours(group)
        if d > 0:
            durations.append(d)

    if not durations:
        return 0.0

    return float(pd.Series(durations).median())


def _std_case_duration_hours(df: pd.DataFrame) -> float:
    if CASE_COL not in df.columns or TIME_COL not in df.columns:
        return 0.0

    durations: list[float] = []
    for _, group in df.groupby(CASE_COL):
        d = _case_duration_hours(group)
        if d > 0:
            durations.append(d)

    if len(durations) < 2:
        return 0.0

    return float(pd.Series(durations).std(ddof=0))


def _safe_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def get_event_durations(case_df: pd.DataFrame, full_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Compute per-case duration evidence for the demo investigator.

    Returns:
    - total duration
    - expected normal duration from the full log if available
    - z-score against the log-level duration std
    - the deviation point, defined as the event immediately before the largest gap
    - remaining activities from the deviation point onward
    """
    if TIME_COL not in case_df.columns or ACTIVITY_COL not in case_df.columns:
        raise KeyError(f"Missing required columns in case_df: {TIME_COL}, {ACTIVITY_COL}")

    case_df = case_df.sort_values(TIME_COL).reset_index(drop=True)
    timestamps = _to_datetime(case_df[TIME_COL])

    if timestamps.isna().all():
        raise ValueError("No valid timestamps found in case_df")

    total_duration_hrs = (timestamps.max() - timestamps.min()).total_seconds() / 3600.0

    gaps: list[float] = []
    for i in range(1, len(timestamps)):
        t_prev = timestamps.iloc[i - 1]
        t_curr = timestamps.iloc[i]
        if pd.isna(t_prev) or pd.isna(t_curr):
            continue
        gap = (t_curr - t_prev).total_seconds() / 3600.0
        gaps.append(max(0.0, float(gap)))

    max_gap_hrs = max(gaps) if gaps else 0.0

    if full_df is not None:
        normal_expected_duration_hrs = _median_case_duration_hours(full_df)
        duration_std = _std_case_duration_hours(full_df)
    else:
        normal_expected_duration_hrs = total_duration_hrs
        duration_std = 0.0

    z_score = 0.0
    if duration_std > 0:
        z_score = (total_duration_hrs - normal_expected_duration_hrs) / duration_std

    is_deviating = bool(
        total_duration_hrs > normal_expected_duration_hrs * 1.5
        or max_gap_hrs > 24.0
    )

    deviation_index = 0
    if gaps:
        deviation_index = int(max(range(len(gaps)), key=lambda i: gaps[i]))

    deviation_index = max(0, min(deviation_index, len(case_df) - 1))
    deviating_activity = str(case_df.iloc[deviation_index][ACTIVITY_COL])
    deviation_timestamp = str(case_df.iloc[deviation_index][TIME_COL])

    remaining_activities = case_df.iloc[deviation_index:][ACTIVITY_COL].astype(str).tolist()

    return {
        "total_duration_hrs": round(float(total_duration_hrs), 4),
        "normal_expected_duration_hrs": round(float(normal_expected_duration_hrs), 4),
        "normal_duration_std_hrs": round(float(duration_std), 4),
        "max_gap_hrs": round(float(max_gap_hrs), 4),
        "z_score": round(float(z_score), 4),
        "is_deviating": is_deviating,
        "deviating_activity": deviating_activity,
        "deviation_timestamp": deviation_timestamp,
        "remaining_activities": remaining_activities,
        "event_gaps_hrs": [round(float(x), 4) for x in gaps],
    }


def get_process_context(df: pd.DataFrame, timestamp: str) -> dict[str, Any]:
    """
    Snapshot the process context at a given timestamp.

    The counts are demo proxies:
    - active cases: cases that have started and not yet ended by timestamp
    - overdue cases: active cases whose elapsed time exceeds the log median case duration
    - near deadline: active cases with <= 24h remaining
    - data quality issues: missing activity/resource values in the preceding 24h
    - avg queue wait: average inter-event gap inside active cases
    - resource utilization: active-case counts by most recent known resource at timestamp
    """
    if CASE_COL not in df.columns or TIME_COL not in df.columns:
        raise KeyError(f"Missing required columns in df: {CASE_COL}, {TIME_COL}")

    work_df = df.copy()
    work_df[TIME_COL] = _to_datetime(work_df[TIME_COL])
    work_df = work_df.dropna(subset=[CASE_COL, TIME_COL])

    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {timestamp}")

    case_bounds = work_df.groupby(CASE_COL)[TIME_COL].agg(case_start="min", case_end="max")
    case_bounds["duration_hrs"] = (
        (case_bounds["case_end"] - case_bounds["case_start"]).dt.total_seconds() / 3600.0
    )

    median_duration = float(case_bounds["duration_hrs"].median()) if not case_bounds.empty else 0.0

    active_mask = (case_bounds["case_start"] <= ts) & (case_bounds["case_end"] >= ts)
    active_cases_df = case_bounds[active_mask]
    active_case_ids = active_cases_df.index.tolist()

    active_cases_count = int(len(active_case_ids))

    elapsed_hrs = (ts - active_cases_df["case_start"]).dt.total_seconds() / 3600.0
    remaining_hrs = (active_cases_df["case_end"] - ts).dt.total_seconds() / 3600.0

    overdue_cases_count = int((elapsed_hrs > median_duration).sum()) if median_duration > 0 else 0
    near_deadline_cases_count = int(((remaining_hrs > 0) & (remaining_hrs <= 24)).sum())

    # data quality issues in the preceding 24 hours
    window_start = ts - pd.Timedelta(hours=24)
    window_df = work_df[(work_df[TIME_COL] >= window_start) & (work_df[TIME_COL] <= ts)]

    data_quality_issues = 0
    if not window_df.empty:
        missing_cols = 0
        if RESOURCE_COL in window_df.columns:
            missing_cols += int(window_df[RESOURCE_COL].isna().sum())
        if ACTIVITY_COL in window_df.columns:
            missing_cols += int(window_df[ACTIVITY_COL].isna().sum())
        data_quality_issues = missing_cols

    # average queue wait proxy:
    # average inter-event gap inside active cases
    queue_wait_values: list[float] = []
    if active_case_ids:
        active_case_events = work_df[work_df[CASE_COL].isin(active_case_ids)]
        for _, case_group in active_case_events.groupby(CASE_COL):
            case_times = case_group[TIME_COL].sort_values().dropna()
            if len(case_times) >= 2:
                diffs = case_times.diff().dt.total_seconds().dropna() / 3600.0
                if not diffs.empty:
                    queue_wait_values.append(float(diffs.mean()))

    avg_queue_wait_hrs = float(pd.Series(queue_wait_values).mean()) if queue_wait_values else 0.0

    # resource utilization proxy:
    # for each active case, take the most recent resource observed at or before ts
    resource_utilization: dict[str, dict[str, Any]] = {}
    if RESOURCE_COL in work_df.columns and active_case_ids:
        snapshot = work_df[
            (work_df[CASE_COL].isin(active_case_ids))
            & (work_df[TIME_COL] <= ts)
        ].sort_values([CASE_COL, TIME_COL])

        latest_resources: list[str] = []
        if not snapshot.empty:
            for _, case_group in snapshot.groupby(CASE_COL):
                last_row = case_group.iloc[-1]
                resource_value = last_row.get(RESOURCE_COL, "UNKNOWN")
                latest_resources.append(str(resource_value) if pd.notna(resource_value) else "UNKNOWN")

        if latest_resources:
            counts = pd.Series(latest_resources).value_counts()
            total_active = max(active_cases_count, 1)
            for resource, count in counts.items():
                resource_utilization[str(resource)] = {
                    "concurrent_cases": int(count),
                    "utilization_pct": round(float(count) / float(total_active), 4),
                }

    return {
        "timestamp": str(ts),
        "active_cases": active_cases_count,
        "overdue_cases": overdue_cases_count,
        "near_deadline_cases": near_deadline_cases_count,
        "data_quality_issues": int(data_quality_issues),
        "avg_queue_wait_hrs": round(float(avg_queue_wait_hrs), 4),
        "resource_utilization": resource_utilization,
    }


def get_affected_cases(
    df: pd.DataFrame,
    resource_id: str,
    anomaly_type: str,
    max_cases: int = 5,
) -> dict[str, Any]:
    """
    Find cases associated with the given resource and return a small, prompt-friendly subset.
    For LateAnomaly, prioritize the longest cases handled by that resource.
    """
    if CASE_COL not in df.columns:
        raise KeyError(f"Missing required column: {CASE_COL}")

    work_df = df.copy()

    if RESOURCE_COL in work_df.columns:
        resource_mask = work_df[RESOURCE_COL].fillna("UNKNOWN").astype(str) == str(resource_id)
        resource_case_ids = work_df.loc[resource_mask, CASE_COL].astype(str).unique().tolist()
    else:
        resource_case_ids = []

    if not resource_case_ids:
        return {
            "resource_id": resource_id,
            "anomaly_type": anomaly_type,
            "count": 0,
            "cases": [],
            "mean_excess_duration_hrs": 0.0,
        }

    if TIME_COL in work_df.columns:
        work_df[TIME_COL] = _to_datetime(work_df[TIME_COL])

    case_durations: dict[str, float] = {}
    for cid, group in work_df[work_df[CASE_COL].isin(resource_case_ids)].groupby(CASE_COL):
        case_durations[str(cid)] = _case_duration_hours(group)

    if anomaly_type == "LateAnomaly" and case_durations:
        expected = float(pd.Series(list(case_durations.values())).median())
        scored = sorted(
            case_durations.items(),
            key=lambda kv: kv[1] - expected,
            reverse=True,
        )
        selected = [cid for cid, _ in scored[:max_cases]]
        excess_values = [max(0.0, case_durations[cid] - expected) for cid in selected]
        mean_excess = float(pd.Series(excess_values).mean()) if excess_values else 0.0
    else:
        selected = list(case_durations.keys())[:max_cases]
        mean_excess = 0.0

    return {
        "resource_id": resource_id,
        "anomaly_type": anomaly_type,
        "count": int(len(case_durations)),
        "cases": selected,
        "mean_excess_duration_hrs": round(float(mean_excess), 4),
    }