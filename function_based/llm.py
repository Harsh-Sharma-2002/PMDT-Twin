from __future__ import annotations

import os
from typing import Any

import ollama


MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def query_llm(prompt: str) -> str:
    """
    Single-pass LLM call for the function-based pipeline.
    Returns raw text so downstream JSON parsing can handle cleanup.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)

        response: dict[str, Any] = client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Always return valid JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9,
            },
        )

        content = response.get("message", {}).get("content", "")
        return str(content).strip()

    except Exception as e:
        return f'{{"error": "{str(e)}"}}'