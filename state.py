from dataclasses import dataclass, field  ### tool calling fix ti functionality 
                                            ### qwen 3.5 testing 2b
from typing import Optional, Any


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
    causal_factor: str  # resource_bottleneck | data_error | deadline_pressure | policy_violation | unknown
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
    # ===== Input =====
    alert: AlertPayload

    # ===== Intermediate Data (PMDT Tool Outputs) =====
    case_info: Optional[dict[str, Any]] = None
    process_context: Optional[dict[str, Any]] = None
    deviation_timestamp: Optional[str] = None
    impacted_cases: list[str] = field(default_factory=list)

    # ===== ReAct / LLM Interaction =====
    messages: list[str] = field(default_factory=list)   # conversation history
    prompt: Optional[str] = None
    final_answer: Optional[str] = None

    investigator_output: Optional[InvestigatorOutput] = None

    # ===== Debug / Trace =====
    tool_call_trace: list[dict[str, Any]] = field(default_factory=list)

    # ✅ NEW: LLM reasoning trace
    llm_trace: list[dict[str, Any]] = field(default_factory=list)

    # ===== Error Handling =====
    error: Optional[str] = None
    failed_node: Optional[str] = None

    # =========================
    # Methods
    # =========================
    def add_trace(self, tool_name: str, tool_input: Any, tool_output: Any) -> None:
        """
        Record tool usage for debugging and evaluation
        """
        self.tool_call_trace.append({
            "tool": tool_name,
            "input": tool_input,
            "output": tool_output,
        })

    # ✅ NEW: LLM trace logger
    def add_llm_trace(self, step_type: str, content: Any) -> None:
        """
        Record LLM reasoning steps (Thought, Action, Observation, Final Answer)
        """
        self.llm_trace.append({
            "type": step_type,
            "content": content
        })

    def add_message(self, content: str) -> None:
        """
        Append to LLM conversation (ReAct loop)
        """
        self.messages.append(content)

    def mark_error(self, node_name: str, error_msg: str) -> None:
        """
        Mark pipeline failure
        """
        self.failed_node = node_name
        self.error = error_msg