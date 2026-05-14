from __future__ import annotations

from typing import Any

import pandas as pd


CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_alert_payload(
    case_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    anomaly_type: str,
    token_replay_fitness: float,
    deviating_activity: str | None = None,
) -> dict[str, Any]:
    """
    Build an AlertPayload matching the benchmark schema.

    Schema:
    - case_id
    - anomaly_type
    - timestamp
    - token_replay_fitness
    - deviating_activity
    - resource_id
    - current_workload
    - is_available
    - shift_end_in_hrs
    """
    if case_df.empty:
        raise ValueError("case_df is empty")

    required_cols = [CASE_COL, TIME_COL, ACTIVITY_COL]
    for col in required_cols:
        if col not in case_df.columns:
            raise KeyError(f"Missing required column in case_df: {col}")

    if TIME_COL not in full_df.columns:
        raise KeyError(f"Missing required column in full_df: {TIME_COL}")

    work_case_df = case_df.copy()
    work_case_df[TIME_COL] = _to_datetime(work_case_df[TIME_COL])
    work_case_df = work_case_df.sort_values(TIME_COL).reset_index(drop=True)

    if work_case_df[TIME_COL].isna().all():
        raise ValueError("No valid timestamps found in case_df")

    # Core identifiers
    case_id = str(work_case_df.iloc[0][CASE_COL])
    case_start_ts = work_case_df[TIME_COL].min()

    # Use provided deviating_activity if available, otherwise fall back to the first activity
    if deviating_activity is not None:
        final_deviating_activity = str(deviating_activity)
    else:
        final_deviating_activity = str(work_case_df.iloc[0][ACTIVITY_COL])

    # Locate the deviation row
    dev_rows = work_case_df[
        work_case_df[ACTIVITY_COL].astype(str) == final_deviating_activity
    ]

    if dev_rows.empty:
        dev_row = work_case_df.iloc[0]
    else:
        dev_row = dev_rows.iloc[0]

    deviation_timestamp = pd.to_datetime(dev_row[TIME_COL], utc=True, errors="coerce")
    if pd.isna(deviation_timestamp):
        deviation_timestamp = case_start_ts

    # Resource handling the case at alert/deviation time
    if RESOURCE_COL in work_case_df.columns:
        resource_value = dev_row.get(RESOURCE_COL, "UNKNOWN")
        resource_id = str(resource_value) if pd.notna(resource_value) else "UNKNOWN"
    else:
        resource_id = "UNKNOWN"

    # Current workload in ±2h window around deviation timestamp
    full_work_df = full_df.copy()
    full_work_df[TIME_COL] = _to_datetime(full_work_df[TIME_COL])
    full_work_df = full_work_df.dropna(subset=[CASE_COL, TIME_COL])

    if RESOURCE_COL in full_work_df.columns:
        window_start = deviation_timestamp - pd.Timedelta(hours=2)
        window_end = deviation_timestamp + pd.Timedelta(hours=2)

        workload_df = full_work_df[
            (full_work_df[TIME_COL] >= window_start)
            & (full_work_df[TIME_COL] <= window_end)
            & (full_work_df[RESOURCE_COL].fillna("UNKNOWN").astype(str) == resource_id)
        ]

        current_workload = int(workload_df[CASE_COL].astype(str).nunique())
    else:
        current_workload = 0

    # Schema rule: available if workload < 15
    is_available = bool(current_workload < 15)

   # Hours remaining until case end from the deviation point,
    # clamped to [0.5, 8.0]
    case_end_ts = work_case_df[TIME_COL].max()

    remaining_hrs = float(
    (case_end_ts - deviation_timestamp).total_seconds() / 3600.0
)

    shift_end_in_hrs = max(
        0.5,
        min(remaining_hrs, 8.0),
    )

    return {
        "case_id": case_id,
        "anomaly_type": str(anomaly_type),
        "timestamp": case_start_ts.isoformat(),
        "token_replay_fitness": round(float(token_replay_fitness), 4),
        "deviating_activity": final_deviating_activity,
        "resource_id": resource_id,
        "current_workload": int(current_workload),
        "is_available": is_available,
        "shift_end_in_hrs": round(float(shift_end_in_hrs), 4),
    }