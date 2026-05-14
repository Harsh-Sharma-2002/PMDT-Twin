from __future__ import annotations

import json
from typing import Any

from state import State, InvestigatorOutput
from utils import parse_llm_output


def _pretty(value: Any) -> str:
    """
    Render dict/list values as readable JSON for the prompt.
    Falls back to plain string if serialization fails.
    """
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2, ensure_ascii=False)
        return str(value)
    except Exception:
        return str(value)


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _to_str_list(value: Any) -> list[str]:
    return [str(x) for x in _to_list(value)]


def build_prompt(state: State) -> str:
    event_durations = state.event_durations or {}
    process_context = state.process_context or {}
    affected_cases = state.affected_cases or {}

    deviation_timestamp = (
        state.deviation_timestamp
        or event_durations.get("deviation_timestamp")
        or state.alert.timestamp
    )
    deviating_activity = event_durations.get(
        "deviating_activity",
        state.alert.deviating_activity,
    )
    remaining_activities = event_durations.get("remaining_activities", [])

    total_duration = float(event_durations.get("total_duration_hrs", 0.0))
    normal_expected = float(event_durations.get("normal_total_duration_hrs", 0.0))
    estimated_delay = max(0.0, total_duration - normal_expected)

    alert_block = {
        "case_id": state.alert.case_id,
        "anomaly_type": state.alert.anomaly_type,
        "timestamp": state.alert.timestamp,
        "token_replay_fitness": state.alert.token_replay_fitness,
        "deviating_activity": state.alert.deviating_activity,
        "resource_id": state.alert.resource_id,
        "current_workload": state.alert.current_workload,
        "is_available": state.alert.is_available,
        "shift_end_in_hrs": state.alert.shift_end_in_hrs,
    }

    return f"""
You are an Investigator Agent for process-mining anomaly analysis.

Your job is to explain why this anomaly likely happened using only the provided evidence.
Do not guess. Do not force a cause if the evidence is weak.
If the evidence is mixed or insufficient, use "unknown".

You are given:
- an AlertPayload
- event duration evidence
- process context evidence
- affected-case evidence

Use at least two independent numeric signals when possible.
Prefer evidence from the log over speculation.
If the cause is resource-related, use resource utilization and overdue cases.
If the cause is data-related, use attribute errors and related context.
If the cause is deadline-related, use near-deadline pressure and queue delay.

=========================
ALERT
=========================
{_pretty(alert_block)}

=========================
EVENT DURATIONS
=========================
{_pretty(event_durations)}

=========================
PROCESS CONTEXT
=========================
{_pretty(process_context)}

=========================
AFFECTED CASES
=========================
{_pretty(affected_cases)}

=========================
DERIVED CHECKS
=========================
Deviation Timestamp: {deviation_timestamp}
Deviating Activity: {deviating_activity}
Remaining Activities: {_pretty(remaining_activities)}
Estimated Delay (hrs): {estimated_delay}

=========================
GUIDELINES
=========================
1. Use numerical evidence wherever possible.
2. Build an evidence chain with explicit numbers.
3. Distinguish:
   - direct cause
   - contributing factors
   - uncertainty
4. Do not return a confident cause if the evidence is weak.
5. If the evidence does not support a clear cause, use:
   - root_cause = "unknown"
   - causal_factor = "unknown"

Allowed causal_factor values:
- resource_bottleneck
- data_error
- deadline_pressure
- policy_violation
- unknown

=========================
OUTPUT FORMAT
=========================
Return ONLY JSON with these fields:

{{
  "root_cause": "short causal label or unknown",
  "causal_factor": "resource_bottleneck | data_error | deadline_pressure | policy_violation | unknown",
  "bottleneck_resource": "resource id or null",
  "trigger_ids": ["signal_1", "signal_2"],
  "trigger_confidence": {{"signal_1": 0.0, "signal_2": 0.0}},
  "evidence_chain": [
    "step 1 with numbers",
    "step 2 with numbers"
  ],
  "impacted_cases": ["case_1", "case_2"],
  "estimated_delay_hrs": 0.0
}}
""".strip()


def run_investigator(state: State, query_fn) -> State:
    try:
        prompt = build_prompt(state)
        state.prompt = prompt

        response = query_fn(prompt)
        state.final_answer = response
        print("\n[LLM OUTPUT]\n", response)

        parsed = parse_llm_output(response)
        if not isinstance(parsed, dict):
            raise ValueError("LLM output could not be parsed into a JSON object")

        event_durations = state.event_durations or {}
        estimated_delay = max(
            0.0,
            float(event_durations.get("total_duration_hrs", 0.0))
            - float(event_durations.get("normal_total_duration_hrs", 0.0)),
        )

        root_cause = str(parsed.get("root_cause", "unknown")).strip() or "unknown"
        causal_factor = str(parsed.get("causal_factor", "unknown")).strip() or "unknown"

        bottleneck_resource = parsed.get("bottleneck_resource")
        if bottleneck_resource is not None:
            bottleneck_resource = str(bottleneck_resource).strip() or None

        if causal_factor == "resource_bottleneck" and not bottleneck_resource:
            bottleneck_resource = state.alert.resource_id

        trigger_ids = _to_str_list(parsed.get("trigger_ids", []))
        evidence_chain = _to_str_list(parsed.get("evidence_chain", []))

        trigger_confidence = parsed.get("trigger_confidence", {})
        if not isinstance(trigger_confidence, dict):
            trigger_confidence = {}

        impacted_cases = _to_str_list(parsed.get("impacted_cases", []))
        if not impacted_cases and isinstance(state.affected_cases, dict):
            impacted_cases = _to_str_list(state.affected_cases.get("cases", []))

        estimated_delay_hrs = parsed.get("estimated_delay_hrs", estimated_delay)
        try:
            estimated_delay_hrs = float(estimated_delay_hrs)
        except Exception:
            estimated_delay_hrs = float(estimated_delay)

        state.investigator_output = InvestigatorOutput(
            root_cause=root_cause,
            causal_factor=causal_factor,
            bottleneck_resource=bottleneck_resource,
            trigger_ids=trigger_ids,
            trigger_confidence=trigger_confidence,
            evidence_chain=evidence_chain,
            impacted_cases=impacted_cases,
            estimated_delay_hrs=estimated_delay_hrs,
        )

        return state

    except Exception as e:
        state.mark_error("investigator", str(e))
        return state