from __future__ import annotations
from collections import defaultdict
from typing import Any
import pandas as pd


CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
AMOUNT_COL = "case:Amount"

def _safe_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    return [value]


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _case_duration_hours(case_df: pd.DataFrame) -> float:
    if TIME_COL not in case_df.columns:
        return 0.0

    ts = _to_datetime(case_df[TIME_COL]).dropna()
    if len(ts) < 2:
        return 0.0

    return (ts.max() - ts.min()).total_seconds() / 3600.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _build_reference_stats(
    full_df: pd.DataFrame,
    sequence_key: tuple[str, ...],
    case_id: str | None = None,
) -> dict[str, Any]:
    """
    Build reference statistics from cases that share the same activity sequence.
    This is the fallback when a precomputed variant_index is not supplied.
    """
    if CASE_COL not in full_df.columns or TIME_COL not in full_df.columns or ACTIVITY_COL not in full_df.columns:
        return {
            "reference_case_count": 0,
            "normal_total_duration_hrs": 0.0,
            "normal_total_duration_std_hrs": 0.0,
            "activity_gap_stats": {},
        }

    ref_case_durations: list[float] = []
    gap_values_by_activity: dict[str, list[float]] = defaultdict(list)

    for ref_case_id, group in full_df.groupby(CASE_COL):
        ref_case_id_str = str(ref_case_id)
        if case_id is not None and ref_case_id_str == case_id:
            continue

        g = group.sort_values(TIME_COL).reset_index(drop=True)
        if g.empty:
            continue

        ref_sequence = tuple(g[ACTIVITY_COL].astype(str).tolist())
        if ref_sequence != sequence_key:
            continue

        ref_duration = _case_duration_hours(g)
        if ref_duration > 0:
            ref_case_durations.append(ref_duration)

        ref_ts = _to_datetime(g[TIME_COL])
        ref_activities = g[ACTIVITY_COL].astype(str).tolist()

        for i in range(len(ref_ts) - 1):
            t_prev = ref_ts.iloc[i]
            t_curr = ref_ts.iloc[i + 1]
            if pd.isna(t_prev) or pd.isna(t_curr):
                continue

            gap_hrs = max(0.0, (t_curr - t_prev).total_seconds() / 3600.0)
            source_activity = ref_activities[i]
            gap_values_by_activity[source_activity].append(float(gap_hrs))

    activity_gap_stats: dict[str, dict[str, float | int]] = {}
    for activity, values in gap_values_by_activity.items():
        s = pd.Series(values, dtype="float64")
        mean_gap = float(s.mean()) if not s.empty else 0.0
        std_gap = float(s.std(ddof=0)) if len(s) >= 2 else 0.0
        activity_gap_stats[activity] = {
            "mean_gap_hrs": round(mean_gap, 4),
            "std_gap_hrs": round(std_gap, 4),
            "sample_count": int(len(values)),
        }

    normal_total_duration_hrs = float(pd.Series(ref_case_durations).median()) if ref_case_durations else 0.0
    normal_total_duration_std_hrs = float(pd.Series(ref_case_durations).std(ddof=0)) if len(ref_case_durations) >= 2 else 0.0

    return {
        "reference_case_count": int(len(ref_case_durations)),
        "normal_total_duration_hrs": round(normal_total_duration_hrs, 4),
        "normal_total_duration_std_hrs": round(normal_total_duration_std_hrs, 4),
        "activity_gap_stats": activity_gap_stats,
    }

###################################################################################################
###################################################################################################

def get_event_durations(
    case_df: pd.DataFrame,
    full_df: pd.DataFrame | None = None,
    *,
    deviating_activity: str | None = None,
    variant_index: dict[Any, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute evidence for the investigator in a way that matches the doc.

    Returns:
    - inter-event gaps
    - normal mean/std per activity
    - z-scores per gap
    - deviation point aligned to deviating_activity if provided
    - total duration and expected normal total duration
    - remaining activities from the deviation point onward
    """
    if TIME_COL not in case_df.columns or ACTIVITY_COL not in case_df.columns:
        raise KeyError(f"Missing required columns in case_df: {TIME_COL}, {ACTIVITY_COL}")

    case_df = case_df.sort_values(TIME_COL).reset_index(drop=True)
    timestamps = _to_datetime(case_df[TIME_COL])

    if timestamps.isna().all():
        raise ValueError("No valid timestamps found in case_df")

    activities = case_df[ACTIVITY_COL].astype(str).tolist()
    case_id = str(case_df[CASE_COL].iloc[0]) if CASE_COL in case_df.columns and not case_df.empty else None
    sequence_key = tuple(activities)

    total_duration_hrs = float((timestamps.max() - timestamps.min()).total_seconds() / 3600.0)

    event_gaps_hrs: list[float] = []
    gap_records: list[dict[str, Any]] = []
    for i in range(len(case_df) - 1):
        t_prev = timestamps.iloc[i]
        t_curr = timestamps.iloc[i + 1]
        if pd.isna(t_prev) or pd.isna(t_curr):
            gap_hrs = 0.0
        else:
            gap_hrs = max(0.0, float((t_curr - t_prev).total_seconds() / 3600.0))

        source_activity = activities[i]
        target_activity = activities[i + 1]
        event_gaps_hrs.append(round(gap_hrs, 4))
        gap_records.append(
            {
                "source_activity": source_activity,
                "target_activity": target_activity,
                "gap_hrs": round(gap_hrs, 4),
            }
        )

    # Prefer a precomputed variant index if provided; otherwise derive from the log.
    reference_case_count = 0
    normal_total_duration_hrs = total_duration_hrs
    normal_total_duration_std_hrs = 0.0
    activity_gap_stats: dict[str, dict[str, float | int]] = {}

    if variant_index is not None and sequence_key in variant_index:
        variant_entry = variant_index[sequence_key]

        if isinstance(variant_entry, dict):
            reference_case_count = int(variant_entry.get("reference_case_count", variant_entry.get("case_count", 0)) or 0)
            normal_total_duration_hrs = _safe_float(
                variant_entry.get("normal_total_duration_hrs", variant_entry.get("mean_total_duration_hrs")),
                default=total_duration_hrs,
            )
            normal_total_duration_std_hrs = _safe_float(
                variant_entry.get("normal_total_duration_std_hrs", variant_entry.get("std_total_duration_hrs")),
                default=0.0,
            )
            activity_gap_stats = variant_entry.get("activity_gap_stats", {}) or {}
        else:
            # If the entry is just a list/set of reference case IDs, use its size.
            reference_case_count = len(_safe_list(variant_entry))
    elif full_df is not None:
        ref_stats = _build_reference_stats(full_df, sequence_key, case_id=case_id)
        reference_case_count = int(ref_stats["reference_case_count"])
        normal_total_duration_hrs = float(ref_stats["normal_total_duration_hrs"])
        normal_total_duration_std_hrs = float(ref_stats["normal_total_duration_std_hrs"])
        activity_gap_stats = ref_stats["activity_gap_stats"]

    # Compute z-scores for each gap against the matching activity baseline.
    z_scores: list[float] = []
    gap_evidence: list[dict[str, Any]] = []
    for i, gap_hrs in enumerate(event_gaps_hrs):
        source_activity = activities[i]

        stats = activity_gap_stats.get(source_activity, {})
        mean_gap = _safe_float(stats.get("mean_gap_hrs", 0.0))
        std_gap = _safe_float(stats.get("std_gap_hrs", 0.0))

        if std_gap > 0:
            z = (gap_hrs - mean_gap) / std_gap
        else:
            z = 0.0

        z = float(z)
        z_scores.append(round(z, 4))
        gap_evidence.append(
            {
                "activity": source_activity,
                "next_activity": activities[i + 1],
                "gap_hrs": round(gap_hrs, 4),
                "normal_mean_gap_hrs": round(mean_gap, 4),
                "normal_std_gap_hrs": round(std_gap, 4),
                "z_score": round(z, 4),
                "is_deviating": bool(abs(z) >= 2.0),
            }
        )

    # Choose the deviation point.
    # 1) Prefer the alert-provided deviating_activity.
    # 2) Otherwise use the largest absolute z-score.
    # 3) Otherwise fall back to the largest gap.
    deviation_index = 0
    if deviating_activity is not None:
        matches = [i for i, act in enumerate(activities) if act == deviating_activity]
        if matches:
            deviation_index = matches[0]
        elif z_scores:
            deviation_index = int(max(range(len(z_scores)), key=lambda i: abs(z_scores[i])))
        elif event_gaps_hrs:
            deviation_index = int(max(range(len(event_gaps_hrs)), key=lambda i: event_gaps_hrs[i]))
    elif z_scores:
        deviation_index = int(max(range(len(z_scores)), key=lambda i: abs(z_scores[i])))
    elif event_gaps_hrs:
        deviation_index = int(max(range(len(event_gaps_hrs)), key=lambda i: event_gaps_hrs[i]))

    deviation_index = max(0, min(deviation_index, len(case_df) - 1))
    deviation_activity = activities[deviation_index]
    deviation_timestamp = str(case_df.iloc[deviation_index][TIME_COL])

    # Use the current case duration vs. normal total duration as the case-level z-score.
    case_duration_std = normal_total_duration_std_hrs
    if case_duration_std > 0:
        case_level_z = (total_duration_hrs - normal_total_duration_hrs) / case_duration_std
    else:
        case_level_z = 0.0

    max_gap_hrs = max(event_gaps_hrs) if event_gaps_hrs else 0.0
    deviation_gap_hrs = event_gaps_hrs[deviation_index] if deviation_index < len(event_gaps_hrs) else 0.0
    deviation_gap_stats = activity_gap_stats.get(deviation_activity, {})
    deviation_gap_mean = _safe_float(deviation_gap_stats.get("mean_gap_hrs", 0.0))
    deviation_gap_std = _safe_float(deviation_gap_stats.get("std_gap_hrs", 0.0))

    # More faithful deviation rule than the old 1.5x heuristic.
    is_deviating = bool(
        abs(case_level_z) >= 2.0
        or (deviation_gap_std > 0 and deviation_gap_hrs > deviation_gap_mean + 2.0 * deviation_gap_std)
        or max_gap_hrs >= 24.0
    )

    remaining_activities = activities[deviation_index:]

    return {
        "total_duration_hrs": round(total_duration_hrs, 4),
        "normal_total_duration_hrs": round(normal_total_duration_hrs, 4),
        "normal_total_duration_std_hrs": round(normal_total_duration_std_hrs, 4),
        "case_level_z_score": round(float(case_level_z), 4),
        "max_gap_hrs": round(float(max_gap_hrs), 4),
        "deviation_gap_hrs": round(float(deviation_gap_hrs), 4),
        "deviation_index": int(deviation_index),
        "is_deviating": is_deviating,
        "deviating_activity": deviation_activity,
        "deviation_timestamp": deviation_timestamp,
        "remaining_activities": remaining_activities,
        "event_gaps_hrs": event_gaps_hrs,
        "gap_evidence": gap_evidence,
        "activity_gap_stats": activity_gap_stats,
        "reference_case_count": int(reference_case_count),
        "z_scores": z_scores,
    }

###################################################################################################
###################################################################################################

def get_process_context(
    df: pd.DataFrame,
    timestamp: str,
    *,
    resource_capacity: int = 15,
    availability_threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Snapshot process context at a specific timestamp.

    Returns investigator-facing evidence signals:
    - active case count
    - overdue case count
    - cases approaching SLA deadline
    - data quality incidents in preceding 24h
    - average queue wait time
    - per-resource concurrent workload + utilization
    - availability flag based on utilization threshold
    """
    if CASE_COL not in df.columns or TIME_COL not in df.columns:
        raise KeyError(f"Missing required columns in df: {CASE_COL}, {TIME_COL}")

    if resource_capacity <= 0:
        raise ValueError("resource_capacity must be positive")

    if not (0.0 < availability_threshold <= 1.0):
        raise ValueError("availability_threshold must be in (0, 1]")

    work_df = df.copy()
    work_df[TIME_COL] = _to_datetime(work_df[TIME_COL])
    work_df = work_df.dropna(subset=[CASE_COL, TIME_COL])

    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {timestamp}")

    # ------------------------------------------------------------------
    # Case lifecycle bounds
    # ------------------------------------------------------------------

    case_bounds = work_df.groupby(CASE_COL)[TIME_COL].agg(
        case_start="min",
        case_end="max",
    )

    case_bounds["duration_hrs"] = (
        (case_bounds["case_end"] - case_bounds["case_start"])
        .dt.total_seconds() / 3600.0
    )

    median_duration_hrs = (
        float(case_bounds["duration_hrs"].median())
        if not case_bounds.empty
        else 0.0
    )

    # ------------------------------------------------------------------
    # Active cases at timestamp
    # ------------------------------------------------------------------

    active_mask = (
        (case_bounds["case_start"] <= ts)
        & (case_bounds["case_end"] >= ts)
    )

    active_cases_df = case_bounds[active_mask]
    active_case_ids = active_cases_df.index.astype(str).tolist()
    active_cases_count = int(len(active_case_ids))

    # ------------------------------------------------------------------
    # Overdue and near-deadline analysis
    # ------------------------------------------------------------------

    elapsed_hrs = (
        (ts - active_cases_df["case_start"])
        .dt.total_seconds() / 3600.0
    )

    remaining_hrs = (
        (active_cases_df["case_end"] - ts)
        .dt.total_seconds() / 3600.0
    )

    overdue_cases_count = (
        int((elapsed_hrs > median_duration_hrs).sum())
        if median_duration_hrs > 0
        else 0
    )

    near_deadline_cases_count = int(
        ((remaining_hrs > 0) & (remaining_hrs <= 24)).sum()
    )

    # ------------------------------------------------------------------
    # Data quality / attribute incidents in the preceding 24 hours
    # ------------------------------------------------------------------

    window_start = ts - pd.Timedelta(hours=24)
    window_df = work_df[
        (work_df[TIME_COL] >= window_start)
        & (work_df[TIME_COL] <= ts)
    ]

    attribute_error_events = 0
    if not window_df.empty:
        if ACTIVITY_COL in window_df.columns:
            attribute_error_events += int(window_df[ACTIVITY_COL].isna().sum())
        if RESOURCE_COL in window_df.columns:
            attribute_error_events += int(window_df[RESOURCE_COL].isna().sum())

    # ------------------------------------------------------------------
    # Average queue wait
    # ------------------------------------------------------------------

    queue_wait_values: list[float] = []

    if active_case_ids:
        active_case_events = work_df[
            work_df[CASE_COL].astype(str).isin(active_case_ids)
        ]

        for _, case_group in active_case_events.groupby(CASE_COL):
            case_times = case_group[TIME_COL].sort_values().dropna()

            if len(case_times) < 2:
                continue

            diffs = (
                case_times.diff()
                .dt.total_seconds()
                .dropna() / 3600.0
            )

            if not diffs.empty:
                queue_wait_values.append(float(diffs.mean()))

    avg_queue_wait_hrs = (
        float(pd.Series(queue_wait_values).mean())
        if queue_wait_values
        else 0.0
    )

    # ------------------------------------------------------------------
    # Resource workload snapshot
    # ------------------------------------------------------------------

    resource_details: dict[str, dict[str, Any]] = {}

    if RESOURCE_COL in work_df.columns and active_case_ids:
        snapshot = work_df[
            (work_df[CASE_COL].astype(str).isin(active_case_ids))
            & (work_df[TIME_COL] <= ts)
        ].sort_values([CASE_COL, TIME_COL])

        latest_resources: list[str] = []

        if not snapshot.empty:
            for _, case_group in snapshot.groupby(CASE_COL):
                last_row = case_group.iloc[-1]
                resource_value = last_row.get(RESOURCE_COL, "UNKNOWN")
                resource_name = (
                    str(resource_value)
                    if pd.notna(resource_value)
                    else "UNKNOWN"
                )
                latest_resources.append(resource_name)

        if latest_resources:
            counts = pd.Series(latest_resources).value_counts()

            for resource, count in counts.items():
                utilization_pct = min(
                    float(count) / float(resource_capacity),
                    1.0,
                )
                is_available = bool(utilization_pct < availability_threshold)

                resource_details[str(resource)] = {
                    "concurrent_cases": int(count),
                    "utilization_pct": round(utilization_pct, 4),
                    "is_available": is_available,
                }

    # ------------------------------------------------------------------
    # Bottleneck signals
    # ------------------------------------------------------------------

    overloaded_resources = [
        resource
        for resource, details in resource_details.items()
        if not details.get("is_available", True)
    ]

    has_resource_bottleneck = bool(
        overloaded_resources and overdue_cases_count > 0
    )

    has_deadline_pressure = bool(near_deadline_cases_count > 0)
    has_data_quality_issue = bool(attribute_error_events > 0)

    return {
        "timestamp": str(ts),

        # Core investigator signals
        "active_cases_count": active_cases_count,
        "overdue_cases_count": overdue_cases_count,
        "near_deadline_cases_count": near_deadline_cases_count,
        "attribute_error_events": int(attribute_error_events),
        "avg_queue_wait_hrs": round(float(avg_queue_wait_hrs), 4),

        # Resource evidence
        "resource_details": resource_details,
        "overloaded_resources": overloaded_resources,

        # Investigator-friendly causal flags
        "has_resource_bottleneck": has_resource_bottleneck,
        "has_deadline_pressure": has_deadline_pressure,
        "has_data_quality_issue": has_data_quality_issue,

        # Baseline context
        "median_case_duration_hrs": round(float(median_duration_hrs), 4),

        # Capacity assumptions used by the surrogate
        "resource_capacity": int(resource_capacity),
        "availability_threshold": round(float(availability_threshold), 4),
    }

###################################################################################################
###################################################################################################

def get_affected_cases(
    df: pd.DataFrame,
    resource_id: str,
    anomaly_type: str,
    anomaly_index: dict[str, list[str]] | None = None,
    max_cases: int = 5,
) -> dict[str, Any]:
    """
    Return case IDs that match the anomaly label and were handled by the given resource.

    Behavior:
    - First filter by anomaly_type using anomaly_index if available.
    - Then keep only cases where resource_id appears in any event's resource field.
    - Return up to max_cases matching case IDs.
    """
    if CASE_COL not in df.columns:
        raise KeyError(f"Missing required column: {CASE_COL}")

    work_df = df.copy()

    # Ensure timestamps are comparable if later sorting is needed
    if TIME_COL in work_df.columns:
        work_df[TIME_COL] = _to_datetime(work_df[TIME_COL])

    # ------------------------------------------------------------
    # Step 1: get candidate cases from anomaly label
    # ------------------------------------------------------------
    if anomaly_index is not None:
        candidate_case_ids = _safe_list(anomaly_index.get(anomaly_type, []))
        candidate_case_ids = [str(cid) for cid in candidate_case_ids]
    else:
        # Fallback: if no anomaly index is provided, use all cases.
        candidate_case_ids = work_df[CASE_COL].astype(str).unique().tolist()

    if not candidate_case_ids:
        return {
            "resource_id": resource_id,
            "anomaly_type": anomaly_type,
            "count": 0,
            "cases": [],
            "mean_excess_duration_hrs": 0.0,
        }

    # ------------------------------------------------------------
    # Step 2: filter to cases where resource_id appears in any event
    # ------------------------------------------------------------
    if RESOURCE_COL not in work_df.columns:
        return {
            "resource_id": resource_id,
            "anomaly_type": anomaly_type,
            "count": 0,
            "cases": [],
            "mean_excess_duration_hrs": 0.0,
        }

    candidate_df = work_df[work_df[CASE_COL].astype(str).isin(candidate_case_ids)].copy()

    if candidate_df.empty:
        return {
            "resource_id": resource_id,
            "anomaly_type": anomaly_type,
            "count": 0,
            "cases": [],
            "mean_excess_duration_hrs": 0.0,
        }

    resource_mask = (
        candidate_df[RESOURCE_COL].fillna("UNKNOWN").astype(str) == str(resource_id)
    )

    matched_case_ids = (
        candidate_df.loc[resource_mask, CASE_COL]
        .astype(str)
        .unique()
        .tolist()
    )

    if not matched_case_ids:
        return {
            "resource_id": resource_id,
            "anomaly_type": anomaly_type,
            "count": 0,
            "cases": [],
            "mean_excess_duration_hrs": 0.0,
        }

    # ------------------------------------------------------------
    # Step 3: compute a small duration summary for prompt context
    # ------------------------------------------------------------
    case_durations: dict[str, float] = {}
    for cid, group in candidate_df[candidate_df[CASE_COL].astype(str).isin(matched_case_ids)].groupby(CASE_COL):
        case_durations[str(cid)] = _case_duration_hours(group)

    selected = matched_case_ids[:max_cases]

    # Mean excess duration relative to the matched affected population
    mean_excess = 0.0
    if case_durations:
        duration_series = pd.Series(list(case_durations.values()), dtype="float64")
        baseline = float(duration_series.median())
        excess_values = [max(0.0, case_durations[cid] - baseline) for cid in selected if cid in case_durations]
        mean_excess = float(pd.Series(excess_values).mean()) if excess_values else 0.0

    return {
        "resource_id": resource_id,
        "anomaly_type": anomaly_type,
        "count": int(len(matched_case_ids)),
        "cases": selected,
        "mean_excess_duration_hrs": round(float(mean_excess), 4),
    }

###################################################################################################
###################################################################################################