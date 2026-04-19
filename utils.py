import json
import re


def parse_llm_output(text: str) -> dict:
    """
    Extract JSON from LLM output.
    Handles messy outputs from small/local models.
    """

    print("\n[RAW LLM OUTPUT]\n", text)

    try:
        # Extract JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            json_str = match.group()

            # Try parsing
            return json.loads(json_str)

        else:
            raise ValueError("No JSON found")

    except Exception as e:
        print("⚠️ JSON parsing failed:", str(e))

        # fallback safe output
        return {
            "root_cause": "unknown",
            "causal_factor": "unknown",
            "bottleneck_resource": None,
            "trigger_ids": [],
            "trigger_confidence": {},
            "evidence_chain": ["Fallback: parsing failed"],
            "estimated_delay_hrs": 0.0
        }