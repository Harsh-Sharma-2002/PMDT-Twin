import ollama


MODEL_NAME = "qwen2.5:3b"   # switch to 7b later if stable


def query_llm(prompt: str) -> str:
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"