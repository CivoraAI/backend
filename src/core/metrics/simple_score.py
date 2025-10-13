def compute_simple_score(text: str) -> float:
    """
    Placeholder metric. Returns a fake score based on text length.
    Replace with real civai_bias pipeline later.
    """
    if not text:
        return 0.0
    # silly heuristic just so we can see numbers move
    return min(1.0, len(text) / 1000.0)
