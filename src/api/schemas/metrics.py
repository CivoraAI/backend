from typing import List

from pydantic import BaseModel, Field


class MetricsRequest(BaseModel):
    articles: List[str] = Field(..., description="Array of article texts to score")


class MetricsResponse(BaseModel):
    scores: List[float]
    overall: float
