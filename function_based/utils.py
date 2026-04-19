from __future__ import annotations

import json
import re
from typing import Any


def _to_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_llm_output(text: str) -> dict:
    """
    Extract and clean JSON from LLM output.
    Designed to handle messy local model responses.
    """
    print("\n[RAW LLM OUTPUT]\n", text)

    try:
        # Extract the first JSON object in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")

        json_str = match.group().strip()

        # Normalize common LLM issues
        json_str = json_str.replace("True", "true")
        json_str = json_str.replace("False", "false")
        json_str = json_str.replace("None", "null")

        # Remove trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        parsed = json.loads(json_str)

        return {
            "root_cause": str(parsed.get("root_cause", "unknown")),
            "causal_factor": str(parsed.get("causal_factor", "unknown")),
            "bottleneck_resource": parsed.get("bottleneck_resource"),
            "trigger_ids": [str(x) for x in _to_list(parsed.get("trigger_ids", []))],
            "trigger_confidence": parsed.get("trigger_confidence") or {},
            "evidence_chain": [str(x) for x in _to_list(parsed.get("evidence_chain", []))],
            "estimated_delay_hrs": _safe_float(parsed.get("estimated_delay_hrs", 0.0)),
        }

    except Exception as e:
        print("⚠️ JSON parsing failed:", str(e))

        return {
            "root_cause": "unknown",
            "causal_factor": "unknown",
            "bottleneck_resource": None,
            "trigger_ids": [],
            "trigger_confidence": {},
            "evidence_chain": ["Fallback: parsing failed"],
            "estimated_delay_hrs": 0.0,
        }
    



    """
    =========================
DERIVED CHECKS
=========================
Deviation Timestamp: 2018-09-12 11:39:31+00:00
Deviating Activity: Declaration REJECTED by SUPERVISOR
Remaining Activities: [
  "Declaration REJECTED by SUPERVISOR",
  "Declaration REJECTED by EMPLOYEE",
  "Declaration SUBMITTED by EMPLOYEE",
  "Declaration REJECTED by ADMINISTRATION",
  "Declaration REJECTED by EMPLOYEE",
  "Declaration SUBMITTED by EMPLOYEE",
  "Declaration REJECTED by ADMINISTRATION",
  "Declaration REJECTED by EMPLOYEE",
  "Declaration SUBMITTED by EMPLOYEE",
  "Declaration APPROVED by ADMINISTRATION",
  "Declaration FINAL_APPROVED by SUPERVISOR",
  "Request Payment",
  "Payment Handled"
]
Estimated Delay (hrs): 472.1295
    
event with timestamps

time stamp based sort

columns meaning exactly

paste the llm 

case examples


    """