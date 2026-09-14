def estimate_tokens(text: str) -> int:
    return max(len(text) // 4, 1)


def truncate_tokens(text: str, max_tokens: int) -> str:
    approx_chars = max_tokens * 4
    return text[:approx_chars]
