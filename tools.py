from typing import Any, Dict


class PMDTTools:
    def get_event_durations(self, case_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_process_context(self, timestamp: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_affected_cases(self, resource_id: str, anomaly_type: str) -> Dict[str, Any]:
        raise NotImplementedError


class MockPMDTTools(PMDTTools):

    def get_event_durations(self, case_id: str) -> Dict[str, Any]:
        return {
            "case_id": case_id,
            "total_duration_hrs": 120,
            "normal_expected_duration_hrs": 48,
            "max_gap_hrs": 36,
            "z_score": 2.8,
            "is_deviating": True
        }

    def get_process_context(self, timestamp: str) -> Dict[str, Any]:
        return {
            "timestamp": timestamp,
            "active_cases": 120,
            "overdue_cases": 30,
            "avg_queue_wait_hrs": 18,
            "resource_utilization": 0.85,
            "data_quality_issues": 2
        }

    def get_affected_cases(self, resource_id: str, anomaly_type: str) -> Dict[str, Any]:
        return {
            "cases": ["CASE_101", "CASE_102", "CASE_103"],
            "count": 3
        }


def get_tool_registry(tools: PMDTTools):
    return {
        "get_event_durations": tools.get_event_durations,
        "get_process_context": tools.get_process_context,
        "get_affected_cases": tools.get_affected_cases,
    }