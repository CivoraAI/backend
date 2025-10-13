from typing import Dict, List

from src.core.metrics.simple_score import compute_simple_score


def score_articles(articles: List[str]) -> Dict:
    """
    Orchestrates scoring multiple articles and returns a summary.
    """
    scores = [compute_simple_score(a) for a in articles]
    overall = sum(scores) / len(scores) if scores else 0.0
    return {"scores": scores, "overall": overall}
