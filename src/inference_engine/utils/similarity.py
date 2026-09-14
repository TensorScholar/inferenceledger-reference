import math
from collections.abc import Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    dot = sum(x * y for x, y in zip(a_list, b_list, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a_list))
    norm_b = math.sqrt(sum(y * y for y in b_list))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
