from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class ScoreItem(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class LLMScorecard(BaseModel):
    market: ScoreItem
    technology: ScoreItem
    competitiveness: ScoreItem
    traction: ScoreItem


class WeightedScore(BaseModel):
    market: float
    technology: float
    competitiveness: float
    traction: float
    total: float


class InvestmentDecision(BaseModel):
    scorecard: LLMScorecard
    weighted_score: WeightedScore
    verdict: Literal["적극 투자", "투자 가능", "보류", "투자 위험"]
    requires_web_search: bool
    decision_rationale: List[str] = Field(default_factory=list)
    missing_information: List[str] = Field(default_factory=list)

