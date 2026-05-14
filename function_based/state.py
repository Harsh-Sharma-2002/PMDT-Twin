from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# =========================
# Alert Payload (Input)
# =========================
@dataclass
class AlertPayload:
    case_id: str
    anomaly_type: str
    timestamp: str
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
# Workflow State
# =========================
@dataclass
class State:
    # -------------------------------------------------
    # Input / benchmark setup
    # -------------------------------------------------
    alert: AlertPayload
    full_df: Optional[pd.DataFrame] = None
    case_df: Optional[pd.DataFrame] = None

    # Precomputed lookup structures
    anomaly_index: Optional[dict[str, list[str]]] = None
    variant_index: Optional[dict[Any, Any]] = None

    # -------------------------------------------------
    # Derived alert context
    # -------------------------------------------------
    deviation_timestamp: Optional[str] = None
    event_durations: Optional[dict[str, Any]] = None
    process_context: Optional[dict[str, Any]] = None
    affected_cases: Optional[dict[str, Any]] = None

    # -------------------------------------------------
    # Prompt / outputs
    # -------------------------------------------------
    prompt: Optional[str] = None
    final_answer: Optional[str] = None
    investigator_output: Optional[InvestigatorOutput] = None

    # -------------------------------------------------
    # Debug / trace
    # -------------------------------------------------
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0

    # -------------------------------------------------
    # Error handling
    # -------------------------------------------------
    error: Optional[str] = None
    failed_node: Optional[str] = None

    def add_trace(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """
        Record a function/tool execution in the benchmark trace.
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