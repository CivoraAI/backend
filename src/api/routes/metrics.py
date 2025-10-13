from fastapi import APIRouter

from src.api.schemas.metrics import MetricsRequest, MetricsResponse
from src.services.metrics_service import score_articles

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/score", response_model=MetricsResponse)
def score(req: MetricsRequest) -> MetricsResponse:
    result = score_articles(req.articles)
    return MetricsResponse(**result)
