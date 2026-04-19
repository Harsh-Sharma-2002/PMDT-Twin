from state import State, InvestigatorOutput
from tools import get_tool_registry, PMDTTools
from utils import parse_llm_output
import json
import re


# =========================
# PROMPT BUILDER (UNCHANGED)
# =========================
def build_initial_prompt(state: State) -> str:
    return f"""
You are an Investigator Agent.

You MUST use tools before giving final answer.

AVAILABLE TOOLS:
- get_event_durations
- get_process_context
- get_affected_cases

STRICT TOOL RULES:
- get_event_durations → input: {{"case_id": string}}
- get_process_context → input: {{"timestamp": string}}
- get_affected_cases → input: {{"resource_id": string, "anomaly_type": string}}

DO NOT invent parameters.

EXECUTION ORDER:                     ### tool not needed update
1. Call get_event_durations FIRST
2. Then call get_process_context
3. Then give Final Answer

FORMAT STRICTLY:

Thought: reasoning
Action: tool_name
Action Input: {{"key": "value"}}

OR

Final Answer:
{{
  "root_cause": "...",
  "causal_factor": "resource_bottleneck | data_error | deadline_pressure | policy_violation | unknown",
  "bottleneck_resource": "... or null",
  "trigger_ids": [...],
  "trigger_confidence": {{}},
  "evidence_chain": ["step1 with numbers", "step2 with numbers"],
  "estimated_delay_hrs": float
}}

OUTPUT RULES:
- estimated_delay_hrs MUST be a number (example: 72.0)
- DO NOT write expressions like 120 - 48
- causal_factor MUST be one of allowed values
- Use numerical evidence from tools

=== ALERT ===
Case ID: {state.alert.case_id}
Anomaly Type: {state.alert.anomaly_type}
Deviating Activity: {state.alert.deviating_activity}
Resource: {state.alert.resource_id}
Workload: {state.alert.current_workload}

Start reasoning.
"""


# =========================
# ROBUST ACTION PARSER
# =========================
def extract_action(text: str):
    try:
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

        if not action_match or not input_match:
            return None, None

        action = action_match.group(1)
        action_input = json.loads(input_match.group(1))

        return action, action_input

    except:
        return None, None


# =========================
# INVESTIGATOR AGENT (UPDATED)
# =========================
def run_investigator(state: State, tools: PMDTTools, query_fn) -> State:
    try:
        tool_registry = get_tool_registry(tools)

        # Initialize prompt
        prompt = build_initial_prompt(state)
        state.prompt = prompt
        state.add_message(prompt)

        for step in range(5):

            # -------------------------
            # LLM CALL
            # -------------------------
            response = query_fn("\n".join(state.messages))

            # ✅ TRACE: LLM response
            state.add_llm_trace("llm_response", response)

            state.add_message(response)

            print(f"\n[STEP {step}] LLM OUTPUT:\n{response}")

            # -------------------------
            # FINAL ANSWER CHECK
            # -------------------------
            if "Final Answer:" in response:
                final_part = response.split("Final Answer:")[-1]
                state.final_answer = final_part

                # ✅ TRACE: final answer
                state.add_llm_trace("final_answer", final_part)

                parsed = parse_llm_output(final_part)

                state.investigator_output = InvestigatorOutput(
                    root_cause=parsed.get("root_cause", "unknown"),
                    causal_factor=parsed.get("causal_factor", "unknown"),
                    bottleneck_resource=parsed.get("bottleneck_resource"),
                    trigger_ids=parsed.get("trigger_ids", []),
                    trigger_confidence=parsed.get("trigger_confidence", {}),
                    evidence_chain=parsed.get("evidence_chain", []),
                    impacted_cases=state.impacted_cases,
                    estimated_delay_hrs=parsed.get("estimated_delay_hrs", 0.0),
                )

                return state

            # -------------------------
            # ACTION EXECUTION
            # -------------------------
            action, action_input = extract_action(response)

            if action and action in tool_registry:

                # 🔒 enforce valid tool inputs
                if action == "get_event_durations":
                    action_input = {"case_id": state.alert.case_id}

                elif action == "get_process_context":
                    action_input = {"timestamp": state.alert.timestamp}

                elif action == "get_affected_cases":
                    action_input = {
                        "resource_id": state.alert.resource_id,
                        "anomaly_type": state.alert.anomaly_type
                    }

                # ✅ TRACE: action
                state.add_llm_trace("action", {
                    "tool": action,
                    "input": action_input
                })

                tool_fn = tool_registry[action]
                result = tool_fn(**action_input)

                # Store in state
                if action == "get_event_durations":
                    state.case_info = result
                    state.deviation_timestamp = state.alert.timestamp

                elif action == "get_process_context":
                    state.process_context = result

                elif action == "get_affected_cases":
                    state.impacted_cases = result.get("cases", [])

                # Tool trace (existing)
                state.add_trace(action, action_input, result)

                # ✅ TRACE: observation
                state.add_llm_trace("observation", result)

                # Feed back observation
                observation = f"Observation: {result}"
                state.add_message(observation)

            else:
                state.add_message(
                    "Observation: Invalid action. Use correct tool format."
                )

        # -------------------------
        # FAILSAFE
        # -------------------------
        state.mark_error("investigator", "Max iterations reached")
        return state

    except Exception as e:
        state.mark_error("investigator", str(e))
        return state