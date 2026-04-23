from data_loader import (
    load_first_xes_from_zip,
    get_longest_case,
    extract_case_features,
    build_alert_from_case,
)
from state import State
from functions import (
    get_event_durations,
    get_process_context,
    get_affected_cases,
)
from investigator import run_investigator
from llm import query_llm
import pandas as pd

def build_raw_event_rows(case_df):
    case_df = case_df.sort_values("time:timestamp").reset_index(drop=True)

    rows = []

    for _, row in case_df.iterrows():
        clean_row = {}

        for k, v in row.to_dict().items():
            if pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, pd.Timestamp):
                clean_row[k] = v.isoformat()  
            else:
                clean_row[k] = v

        rows.append(clean_row)

    return rows

def main():
    zip_path = "BPI Challenge 2020_ Domestic Declarations_1_all.zip"

    print(f"Loading dataset from: {zip_path}")
    df = load_first_xes_from_zip(zip_path)

    print("Columns:", list(df.columns))

    case_id, case_df = get_longest_case(df)
    print(f"\nSelected case: {case_id}")
    print(case_df.head())

    features = extract_case_features(case_df)
    print("\nCase features:", features)

    # Precompute anomaly evidence first so the alert can use the deviation timestamp
    event_durations = get_event_durations(case_df, df)
    print("\nEvent durations:", event_durations)

    alert = build_alert_from_case(
        case_id=case_id,
        case_df=case_df,
        features=features,
        full_df=df,
        event_durations=event_durations,
    )
    print("\nAlert:", alert)

    state = State(alert=alert)

    # Store precomputed evidence in state
    state.event_durations = event_durations
    state.process_context = get_process_context(df, alert.timestamp)
    state.affected_cases = get_affected_cases(df, alert.resource_id, alert.anomaly_type)

    # Raw XES event logs
    state.raw_event_rows = build_raw_event_rows(case_df)

    # Track precomputed function calls
    state.add_trace(
        "get_event_durations",
        {"case_id": alert.case_id},
        state.event_durations,
    )
    state.add_trace(
        "get_process_context",
        {"timestamp": alert.timestamp},
        state.process_context,
    )
    state.add_trace(
        "get_affected_cases",
        {
            "resource_id": alert.resource_id,
            "anomaly_type": alert.anomaly_type,
        },
        state.affected_cases,
    )

    # Run investigator once with all context
    state = run_investigator(state, query_llm)

    print("\n============================")
    print("FINAL PROMPT (LLM INPUT)")
    print("============================\n")
    print(state.prompt)

    print("\n============================")
    print("FINAL INVESTIGATION RESULT")
    print("============================\n")

    if state.error:
        print("ERROR:", state.error)
        print("Failed Node:", state.failed_node)
        return

    output = state.investigator_output
    if not output:
        print("No output generated")
        return

    what_happened = getattr(output, "what_happened", None)

    print("What Happened:", what_happened if what_happened else "N/A")
    print("Root Cause:", output.root_cause)
    print("Causal Factor:", output.causal_factor)
    print("Bottleneck Resource:", output.bottleneck_resource)
    print("Estimated Delay (hrs):", output.estimated_delay_hrs)

    print("\nTrigger IDs:", output.trigger_ids)
    print("Trigger Confidence:", output.trigger_confidence)

    print("\nEvidence Chain:")
    for step in output.evidence_chain:
        print("-", step)

    print("\nImpacted Cases:", output.impacted_cases)

    print("\n============================")
    print("FUNCTION TRACE")
    print("============================\n")
    for trace in state.tool_call_trace:
        print(f"Function: {trace['tool']}")
        print(f"Input: {trace['input']}")
        print(f"Output: {trace['output']}")
        print("------")


if __name__ == "__main__":
    main()