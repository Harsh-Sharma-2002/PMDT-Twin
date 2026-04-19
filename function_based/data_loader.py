import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import pm4py


CASE_COL = "case:concept:name"
TIME_COL = "time:timestamp"
ACTIVITY_COL = "concept:name"
RESOURCE_COL = "org:resource"
AMOUNT_COL = "case:Amount"


def load_first_xes_from_zip(zip_path: str) -> pd.DataFrame:
    """
    Load the first .xes or .xes.gz file inside a zip archive into a dataframe.
    """
    path = Path(zip_path)
    if not path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        print("Zip contents:")
        for name in names:
            print(" -", name)

        xes_files = [
            name for name in names
            if name.lower().endswith(".xes") or name.lower().endswith(".xes.gz")
        ]

        if not xes_files:
            raise ValueError(f"No .xes or .xes.gz file found inside {zip_path}")

        chosen = xes_files[0]
        print(f"\nLoading file: {chosen}")

        with tempfile.TemporaryDirectory() as tmpdir:
            extracted_path = zf.extract(chosen, path=tmpdir)
            log = pm4py.read_xes(extracted_path)
            df = pm4py.convert_to_dataframe(log)

    return df


def get_longest_case(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Pick the longest case by number of events.
    """
    if CASE_COL not in df.columns:
        raise KeyError(f"Missing column: {CASE_COL}")
    if TIME_COL not in df.columns:
        raise KeyError(f"Missing column: {TIME_COL}")

    case_sizes = df.groupby(CASE_COL).size().sort_values(ascending=False)
    case_id = case_sizes.index[0]
    case_df = df[df[CASE_COL] == case_id].sort_values(TIME_COL)

    return str(case_id), case_df


def _pick_alert_resource(case_df: pd.DataFrame) -> str:
    """
    Pick a reasonable resource for the alert:
    prefer the most frequent non-SYSTEM resource, otherwise the most frequent resource.
    """
    if RESOURCE_COL not in case_df.columns:
        return "UNKNOWN"

    resources = case_df[RESOURCE_COL].fillna("UNKNOWN").astype(str).tolist()
    if not resources:
        return "UNKNOWN"

    filtered = [r for r in resources if r.upper() != "SYSTEM"]
    if filtered:
        return Counter(filtered).most_common(1)[0][0]

    return Counter(resources).most_common(1)[0][0]


def extract_case_features(case_df: pd.DataFrame) -> dict:
    """
    Compute simple case-level features from one real BPIC20 case.

    This is a lightweight descriptive summary used by the demo harness.
    The actual anomaly timestamp should come from event-duration analysis.
    """
    if TIME_COL not in case_df.columns or ACTIVITY_COL not in case_df.columns:
        raise KeyError(f"Missing required columns in case_df: {TIME_COL}, {ACTIVITY_COL}")

    case_df = case_df.sort_values(TIME_COL).reset_index(drop=True)
    timestamps = pd.to_datetime(case_df[TIME_COL], utc=True, errors="coerce")

    if timestamps.isna().all():
        raise ValueError("No valid timestamps found in case_df")

    duration_hours = (timestamps.max() - timestamps.min()).total_seconds() / 3600.0

    if RESOURCE_COL in case_df.columns:
        resources = case_df[RESOURCE_COL].fillna("UNKNOWN").astype(str).tolist()
    else:
        resources = ["UNKNOWN"] * len(case_df)

    reassignments = sum(
        1 for i in range(1, len(resources)) if resources[i] != resources[i - 1]
    )

    activities = case_df[ACTIVITY_COL].astype(str).tolist()
    loops = len(activities) - len(set(activities))

    amount = None
    if AMOUNT_COL in case_df.columns and not case_df[AMOUNT_COL].isna().all():
        amount = float(case_df[AMOUNT_COL].iloc[0])

    return {
        "duration_hours": float(duration_hours),
        "reassignments": int(reassignments),
        "loops": int(loops),
        "amount": amount,
        "start_timestamp": str(timestamps.min()),
        "end_timestamp": str(timestamps.max()),
        "deviating_activity": str(case_df.iloc[-1][ACTIVITY_COL]),
        "resource_id": _pick_alert_resource(case_df),
    }


def _estimate_current_workload(
    df: Optional[pd.DataFrame],
    resource_id: str,
    timestamp: str,
    window_hours: float = 2.0,
) -> int:
    """
    Approximate concurrent workload using cases that have events
    for the same resource in a +/- time window around the timestamp.
    """
    if df is None or RESOURCE_COL not in df.columns or TIME_COL not in df.columns:
        return 0

    work_df = df.copy()
    work_df[TIME_COL] = pd.to_datetime(work_df[TIME_COL], utc=True, errors="coerce")
    work_df = work_df.dropna(subset=[TIME_COL])

    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(ts):
        return 0

    window_start = ts - pd.Timedelta(hours=window_hours)
    window_end = ts + pd.Timedelta(hours=window_hours)

    window_df = work_df[
        (work_df[TIME_COL] >= window_start)
        & (work_df[TIME_COL] <= window_end)
        & (work_df[RESOURCE_COL].fillna("UNKNOWN").astype(str) == str(resource_id))
    ]

    if CASE_COL in window_df.columns:
        return int(window_df[CASE_COL].nunique())

    return int(len(window_df))


def build_alert_from_case(
    case_id: str,
    case_df: pd.DataFrame,
    features: dict,
    full_df: Optional[pd.DataFrame] = None,
    event_durations: Optional[dict] = None,
):
    """
    Build a realistic alert payload from the selected real case.

    Important:
    - timestamp represents the deviation moment
    - deviation_timestamp is explicitly included in the payload
    - deviating activity should come from event-duration analysis when available
    - resource_id is the dominant non-SYSTEM resource in the case
    - current_workload is approximated from the full log if available
    """
    from state import AlertPayload

    case_df = case_df.sort_values(TIME_COL).reset_index(drop=True)
    timestamps = pd.to_datetime(case_df[TIME_COL], utc=True, errors="coerce")
    if timestamps.isna().all():
        raise ValueError("No valid timestamps found in case_df")

    start_timestamp = str(timestamps.iloc[0])
    end_timestamp = str(timestamps.iloc[-1])

    deviating_activity = str(features.get("deviating_activity", case_df.iloc[-1][ACTIVITY_COL]))
    resource_id = str(features.get("resource_id", _pick_alert_resource(case_df)))

    deviation_timestamp = end_timestamp
    if isinstance(event_durations, dict):
        deviation_timestamp = str(event_durations.get("deviation_timestamp", end_timestamp))
        deviating_activity = str(event_durations.get("deviating_activity", deviating_activity))

    current_workload = _estimate_current_workload(
        full_df,
        resource_id,
        deviation_timestamp,
    )
    is_available = current_workload < 15

    duration_hours = float(features.get("duration_hours", 0.0))
    shift_end_in_hrs = min(max(duration_hours / 24.0, 0.5), 8.0)

    anomaly_type = "LateAnomaly" if duration_hours > 24 else "Normal"

    return AlertPayload(
        case_id=str(case_id),
        anomaly_type=anomaly_type,
        timestamp=deviation_timestamp,
        deviation_timestamp=deviation_timestamp,
        token_replay_fitness=0.72,
        deviating_activity=deviating_activity,
        resource_id=resource_id,
        current_workload=current_workload,
        is_available=is_available,
        shift_end_in_hrs=shift_end_in_hrs,
    )