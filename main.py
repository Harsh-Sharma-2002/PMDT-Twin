from state import State, AlertPayload
from tools import MockPMDTTools
from investigator import run_investigator
from llm import query_llm


def main():

    # 1. CREATE ALERT (placeholder)

    alert = AlertPayload(
        case_id="CASE_123",
        anomaly_type="LateAnomaly",
        timestamp="2024-01-01T10:00:00",
        token_replay_fitness=0.72,
        deviating_activity="Approve Request",
        resource_id="USER_A",
        current_workload=12,
        is_available=False,
        shift_end_in_hrs=2.5
    )

    # 2. INIT STATE
   
    state = State(alert=alert)

   
    # 3. INIT TOOLS
   
    tools = MockPMDTTools()

 
    # 4. RUN INVESTIGATOR
   
    state = run_investigator(state, tools, query_llm)

    # =========================
    # PRINT FINAL PROMPT
    # ========================= 
    print("\n============================")
    print("FINAL PROMPT (LLM INPUT)")
    print("============================\n")

    if state.messages:
    # full ReAct conversation
        print("\n--- FULL CONVERSATION ---\n")
        print("\n".join(state.messages))

    elif state.prompt:
        # fallback if not using messages
        print(state.prompt)

    else:
        print("⚠️ No prompt captured")
    # 5. HANDLE RESULT
   
    print("\n============================")
    print("FINAL INVESTIGATION RESULT")
    print("============================\n")

    if state.error:
        print("❌ ERROR:", state.error)
        print("Failed Node:", state.failed_node)
        return

    output = state.investigator_output

    if not output:
        print("⚠️ No output generated")
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

   
    # 6. DEBUG TRACE 
 
    print("\n============================")
    print("TOOL TRACE")
    print("============================\n")

    for trace in state.tool_call_trace:
        print(f"Tool: {trace['tool']}")
        print(f"Input: {trace['input']}")
        print(f"Output: {trace['output']}")
        print("------")


if __name__ == "__main__":
    main()