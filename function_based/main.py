from data_loader import (
    load_first_xes_from_zip,
    get_longest_case,
    extract_case_features,
    build_raw_logs,
)
from payload_builder import build_alert_payload
from state import State
from functions import (
    get_event_durations,
    get_process_context,
    get_affected_cases,
)
from investigator import run_investigator
from llm import query_llm


ZIP_PATH = "BPI Challenge 2020_ Domestic Declarations_1_all.zip"


def main() -> None:
    print(f"Loading dataset from: {ZIP_PATH}")
    df = load_first_xes_from_zip(ZIP_PATH)
    print("Columns:", list(df.columns))

    case_id, case_df = get_longest_case(df)
    print(f"\nSelected case: {case_id}")
    print(case_df.head())

    features = extract_case_features(case_df)
    print("\nCase features:", features)

    # Precompute evidence before building the alert.
    event_durations = get_event_durations(case_df, df)
    print("\nEvent durations:", event_durations)

    alert = build_alert_payload(
        case_id=case_id,
        case_df=case_df,
        features=features,
        full_df=df,
        event_durations=event_durations,
    )
    print("Alert:", alert)

    state = State(alert=alert)
    state.event_durations = event_durations
    state.raw_logs = build_raw_logs(case_df)
    state.process_context = get_process_context(df, alert.timestamp)
    state.affected_cases = get_affected_cases(df, alert.resource_id, alert.anomaly_type)

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
    if output is None:
        print("No output generated")
        return

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
