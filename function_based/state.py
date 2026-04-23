from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# =========================
# Alert Payload (Input)
# =========================
@dataclass
class AlertPayload:
    case_id: str
    anomaly_type: str
    timestamp: str
    deviation_timestamp: str
    token_replay_fitness: float
    deviating_activity: str
    resource_id: str
    current_workload: int
    is_available: bool
    shift_end_in_hrs: float


# =========================
# Investigator Output
# =========================
@dataclass
class InvestigatorOutput:
    root_cause: str
    causal_factor: str
    bottleneck_resource: Optional[str]
    trigger_ids: list[str]
    trigger_confidence: dict[str, float]
    evidence_chain: list[str]
    impacted_cases: list[str]
    estimated_delay_hrs: float


# =========================
# Workflow State (FUNCTION BASED)
# =========================
@dataclass
class State:
    # ===== Input =====
    alert: AlertPayload

    # ===== Raw Evidence =====
    raw_event_rows: Optional[list[dict[str, Any]]] = None

    # ===== Precomputed Signals =====
    event_durations: Optional[dict[str, Any]] = None
    process_context: Optional[dict[str, Any]] = None
    affected_cases: Optional[dict[str, Any]] = None

    # ===== Prompt / Output =====
    prompt: Optional[str] = None
    final_answer: Optional[str] = None
    investigator_output: Optional[InvestigatorOutput] = None

    # ===== Debug =====
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0

    # ===== Error Handling =====
    error: Optional[str] = None
    failed_node: Optional[str] = None

    def add_trace(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """
        Record function execution.
        """
        self.tool_call_trace.append(
            {
                "tool": tool_name,
                "input": tool_input,
                "output": tool_output,
            }
        )
        self.tool_call_count += 1

    def mark_error(self, node_name: str, error_msg: str) -> None:
        self.failed_node = node_name
        self.error = error_msg