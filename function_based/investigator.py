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


def _to_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]



def build_prompt(state: State) -> str:
    raw_events = state.raw_event_rows or []

    return f"""
You are an Investigator Agent.

Your job is to analyze the process execution and infer the most plausible explanation using ONLY the raw event log.

You do NOT have access to:
- alert data
- aggregated statistics
- historical data
- external context

Do NOT invent missing data.

=========================
PROCESS BACKGROUND
=========================
This dataset represents a travel expense declaration process (BPI Challenge 2020).

Typical process flow:
- An employee submits a declaration for reimbursement
- The declaration is reviewed by a supervisor and/or administration
- The declaration may be:
  - approved → moves forward
  - rejected → sent back to employee for correction
  - resubmitted → re-enters review

Important process behaviors:
- A rejection usually indicates missing, incorrect, or non-compliant information
- Multiple rejection–resubmission cycles indicate rework or process inefficiency
- Long gaps between events indicate waiting or delays
- Multiple resource handoffs indicate coordination overhead

=========================
RAW EVENT LOG (PRIMARY EVIDENCE)
=========================
{_pretty(raw_events)}

=========================
GUIDELINES
=========================
1. Use ONLY the raw event log for reasoning.
2. Identify patterns such as:
   - repeated rejection and resubmission (rework)
   - long time gaps (delays)
   - loops or repeated activities
   - unusual ordering of activities
3. Do NOT assume hidden data or external context.
4. Distinguish clearly between:
   - what is observed
   - what is inferred
   - what is unknown

5. If strong evidence exists, infer a plausible cause.
6. Otherwise, return "unknown".
7. Avoid overconfidence.

IMPORTANT:
- Activities are observations, not proof of causality
- Patterns must be supported by the event sequence

=========================
OUTPUT FORMAT
=========================
Return ONLY JSON:

{{
  "what_happened": "Short paragraph explaining what likely happened in this case.",
  "root_cause_explanation": "...",
  "direct_cause": "... or unknown",
  "contributing_factors": ["...", "..."],
  "confidence": 0.0,
  "evidence_chain": ["...", "..."]
}}
""".strip()


########################################################################################


# def build_prompt(state: State) -> str:
#     event_durations = state.event_durations or {}
#     process_context = state.process_context or {}
#     affected_cases = state.affected_cases or {}

#     deviation_timestamp = event_durations.get("deviation_timestamp", state.alert.timestamp)
#     deviating_activity = event_durations.get("deviating_activity", state.alert.deviating_activity)
#     remaining_activities = event_durations.get("remaining_activities", [])

#     total_duration = float(event_durations.get("total_duration_hrs", 0.0))
#     normal_expected = float(event_durations.get("normal_expected_duration_hrs", 0.0))
#     estimated_delay = max(0.0, total_duration - normal_expected)

#     alert_block = {
#         "case_id": state.alert.case_id,
#         "anomaly_type": state.alert.anomaly_type,
#         "timestamp": state.alert.timestamp,
#         "deviating_activity": state.alert.deviating_activity,
#         "resource_id": state.alert.resource_id,
#         "current_workload": state.alert.current_workload,
#         "is_available": state.alert.is_available,
#         "shift_end_in_hrs": state.alert.shift_end_in_hrs,
#     }

#     return f"""
# You are an Investigator Agent.

# Your job is to analyze the anomaly and infer the most plausible explanation using ONLY the alert data.

# You do NOT have access to event logs, process statistics, or historical data.

# However, you are given background knowledge about the process.

# =========================
# PROCESS BACKGROUND
# =========================
# This dataset represents a travel expense declaration process (BPI Challenge 2020).

# Typical process flow:
# - An employee submits a declaration for reimbursement
# - The declaration is reviewed by a supervisor and/or administration
# - The declaration may be:
#   - approved → moves forward
#   - rejected → sent back to employee for correction
#   - resubmitted → re-enters review

# Important process behaviors:
# - A rejection usually indicates missing, incorrect, or non-compliant information
# - Multiple rejection–resubmission cycles indicate rework or process inefficiency
# - A "LateAnomaly" means an activity occurred later than expected in the process timeline
# - Resource workload may affect speed, but does NOT explain why a rejection occurs
# - A rejection may be either:
#   - the cause of delay (due to rework), OR
#   - the result of delay (e.g., deadline violation)

# =========================
# ALERT
# =========================
# {_pretty(alert_block)}

# =========================
# GUIDELINES
# =========================
# 1. Use ONLY the alert data for reasoning.
# 2. You may use the process background to interpret what the alert means.
# 3. Do NOT assume hidden data or external context.
# 4. Distinguish clearly between:
#    - what is observed
#    - what is inferred
#    - what is unknown

# 5. You may infer a plausible cause IF the alert strongly suggests it.
# 6. Otherwise, return "unknown" and explain why.

# 7. Avoid overconfidence. Prefer cautious reasoning.

# IMPORTANT:
# - The deviating activity is an observation, not proof of causality
# - A rejection event suggests possible rework, but does not guarantee it caused the delay
# - Resource availability does NOT explain why a declaration was rejected

# =========================
# OUTPUT FORMAT
# =========================
# Return ONLY JSON:

# {{
#   "what_happened": "Short paragraph explaining what likely happened in this case.",
#   "root_cause_explanation": "...",
#   "direct_cause": "... or unknown",
#   "contributing_factors": ["...", "..."],
#   "confidence": 0.0,
#   "evidence_chain": ["...", "..."]
# }}
# """.strip()

########################################################################################

# def build_prompt(state: State) -> str:
#     event_durations = state.event_durations or {}
#     process_context = state.process_context or {}
#     affected_cases = state.affected_cases or {}

#     deviation_timestamp = event_durations.get("deviation_timestamp", state.alert.timestamp)
#     deviating_activity = event_durations.get("deviating_activity", state.alert.deviating_activity)
#     remaining_activities = event_durations.get("remaining_activities", [])

#     total_duration = float(event_durations.get("total_duration_hrs", 0.0))
#     normal_expected = float(event_durations.get("normal_expected_duration_hrs", 0.0))
#     estimated_delay = max(0.0, total_duration - normal_expected)

#     alert_block = {
#         "case_id": state.alert.case_id,
#         "anomaly_type": state.alert.anomaly_type,
#         "timestamp": state.alert.timestamp,
#         "deviating_activity": state.alert.deviating_activity,
#         "resource_id": state.alert.resource_id,
#         "current_workload": state.alert.current_workload,
#         "is_available": state.alert.is_available,
#         "shift_end_in_hrs": state.alert.shift_end_in_hrs,
#     }

#     return f"""
# You are an Investigator Agent.

# Your job is to analyze the situation and determine the most plausible cause(s) of the anomaly.

# You are given:
# - alert data
# - event duration analysis
# - process context
# - affected cases

# Use all provided data carefully.

# Do NOT assume causes that are not supported by evidence.
# It is completely acceptable to say the cause is uncertain or unknown.

# Focus on:
# - identifying what likely caused the delay
# - distinguishing between direct cause and contributing factors
# - explaining your reasoning using the available signals

# =========================
# ALERT
# =========================
# {_pretty(alert_block)}

# =========================
# EVENT DURATIONS
# =========================
# {_pretty(event_durations)}

# =========================
# PROCESS CONTEXT
# =========================
# {_pretty(process_context)}

# =========================
# AFFECTED CASES
# =========================
# {_pretty(affected_cases)}

# =========================
# DERIVED CHECKS
# =========================
# Deviation Timestamp: {deviation_timestamp}
# Deviating Activity: {deviating_activity}
# Remaining Activities: {_pretty(remaining_activities)}
# Estimated Delay (hrs): {estimated_delay}

# =========================
# GUIDELINES
# =========================
# 1. Use numerical evidence wherever possible.
# 2. Use at least TWO independent signals when available.
# 3. Distinguish between:
#    - direct cause (if identifiable)
#    - contributing factors
#    - uncertainty

# 4. Do NOT force a conclusion if the data does not clearly support one.
# 5. Prefer cautious, evidence-based reasoning over confident guesses.

# IMPORTANT:
# - estimated_delay_hrs = total_duration_hrs - normal_expected_duration_hrs
# - Do NOT write expressions like 120 - 48
# - The deviating activity and timestamp are useful anchors, but do NOT prove causality by themselves

# =========================
# OUTPUT FORMAT
# =========================
# Return ONLY JSON:

# {{
#   "what_happened": "Short paragraph (3–5 sentences) explaining what likely happened in this case in plain language.",
#   "root_cause_explanation": "...",
#   "direct_cause": "... or unknown",
#   "contributing_factors": ["...", "..."],
#   "confidence": 0.0,
#   "evidence_chain": ["...", "..."],
#   "estimated_delay_hrs": 0.0
# }}
# """.strip()

######################################################################


# def build_prompt(state: State) -> str:
#     event_durations = state.event_durations or {}
#     process_context = state.process_context or {}
#     affected_cases = state.affected_cases or {}

#     deviation_timestamp = event_durations.get("deviation_timestamp", state.alert.timestamp)
#     deviating_activity = event_durations.get("deviating_activity", state.alert.deviating_activity)
#     remaining_activities = event_durations.get("remaining_activities", [])

#     total_duration = float(event_durations.get("total_duration_hrs", 0.0))
#     normal_expected = float(event_durations.get("normal_expected_duration_hrs", 0.0))
#     estimated_delay = max(0.0, total_duration - normal_expected)

#     alert_block = {
#         "case_id": state.alert.case_id,
#         "anomaly_type": state.alert.anomaly_type,
#         "timestamp": state.alert.timestamp,
#         "deviating_activity": state.alert.deviating_activity,
#         "resource_id": state.alert.resource_id,
#         "current_workload": state.alert.current_workload,
#         "is_available": state.alert.is_available,
#         "shift_end_in_hrs": state.alert.shift_end_in_hrs,
#     }

#     return f"""
# You are an Investigator Agent.

# Your job is to determine the ROOT CAUSE of an anomaly using structured evidence.

# You are given:
# - alert data
# - event duration analysis
# - process context
# - affected cases

# Use all provided data.
# Prefer numerical evidence over vague reasoning.
# Do not invent facts that are not in the data.

# =========================
# ALERT
# =========================
# {_pretty(alert_block)}

# =========================
# EVENT DURATIONS
# =========================
# {_pretty(event_durations)}

# =========================
# PROCESS CONTEXT
# =========================
# {_pretty(process_context)}

# =========================
# AFFECTED CASES
# =========================
# {_pretty(affected_cases)}

# =========================
# DERIVED CHECKS
# =========================
# Deviation Timestamp: {deviation_timestamp}
# Deviating Activity: {deviating_activity}
# Remaining Activities: {_pretty(remaining_activities)}
# Estimated Delay (hrs): {estimated_delay}

# =========================
# RULES
# =========================
# 1. Use at least TWO independent signals.
# 2. Use numerical evidence.
# 3. Choose ONE causal_factor from:
#    - resource_bottleneck
#    - data_error
#    - deadline_pressure
#    - policy_violation
#    - unknown

# 4. estimated_delay_hrs must be a number.
# 5. Return valid JSON only.
# 6. Use JSON booleans and null only:
#    - true / false / null
#    Do NOT use Python booleans like True / False.

# IMPORTANT:
# - estimated_delay_hrs must be computed as total_duration_hrs - normal_expected_duration_hrs
# - Do NOT write expressions like 120 - 48
# - If causal_factor is resource_bottleneck, bottleneck_resource should usually be the alert resource_id unless the evidence clearly says otherwise
# - Use the deviation timestamp and deviating activity from the event-duration analysis when they are present

# =========================
# OUTPUT FORMAT
# =========================
# Return ONLY JSON:

# {{
#   "root_cause": "...",
#   "causal_factor": "...",
#   "bottleneck_resource": "... or null",
#   "trigger_ids": [...],
#   "trigger_confidence": {{...}},
#   "evidence_chain": ["...", "..."],
#   "estimated_delay_hrs": 0.0
# }}
# """.strip()


def run_investigator(state: State, query_fn) -> State:
    try:
        prompt = build_prompt(state)
        state.prompt = prompt

        response = query_fn(prompt)
        state.final_answer = response
        print("\n[LLM OUTPUT]\n", response)

        parsed = parse_llm_output(response)

        estimated_delay = 0.0
        if isinstance(state.event_durations, dict):
            estimated_delay = max(
                0.0,
                float(state.event_durations.get("total_duration_hrs", 0.0))
                - float(state.event_durations.get("normal_expected_duration_hrs", 0.0)),
            )

        causal_factor = str(parsed.get("causal_factor", "unknown"))
        bottleneck_resource = parsed.get("bottleneck_resource")
        trigger_confidence = parsed.get("trigger_confidence", {})

        if causal_factor == "resource_bottleneck" and not bottleneck_resource:
            bottleneck_resource = state.alert.resource_id

        if trigger_confidence is None or not isinstance(trigger_confidence, dict):
            trigger_confidence = {}

        impacted_cases = []
        if isinstance(state.affected_cases, dict):
            impacted_cases = _to_list(state.affected_cases.get("cases", []))

        trigger_ids = _to_list(parsed.get("trigger_ids", []))
        evidence_chain = _to_list(parsed.get("evidence_chain", []))

        state.investigator_output = InvestigatorOutput(
            root_cause=str(parsed.get("root_cause", "unknown")),
            causal_factor=causal_factor,
            bottleneck_resource=bottleneck_resource,
            trigger_ids=[str(x) for x in trigger_ids],
            trigger_confidence=trigger_confidence,
            evidence_chain=[str(x) for x in evidence_chain],
            impacted_cases=[str(x) for x in impacted_cases],
            estimated_delay_hrs=float(parsed.get("estimated_delay_hrs", estimated_delay)),
        )

        return state

    except Exception as e:
        state.mark_error("investigator", str(e))
        return state
    



    