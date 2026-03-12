from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Reference(BaseModel):
    title: str = Field(description="출처 제목")
    url: str = Field(description="출처 URL")
    source_type: Literal["rag", "web", "manual"] = Field(default="rag")


class StartupProfile(BaseModel):
    company_name: str
    domain: str = "Robotics"
    one_liner: str
    headquarters: Optional[str] = None
    founded_year: Optional[int] = None
    representative: Optional[str] = None
    products: List[str] = Field(default_factory=list)
    customers: List[str] = Field(default_factory=list)
    fundraising_history: List[str] = Field(default_factory=list)
    patents: List[str] = Field(default_factory=list)


class ExplorationSummary(BaseModel):
    competitor_summary: str = Field(description="경쟁사 대비 강점/약점 요약")
    traction_summary: str = Field(description="고객사, 매출, 투자유치 등 실적 요약")
    references: List[Reference] = Field(default_factory=list)


class TechnologySummary(BaseModel):
    core_technology: str
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    differentiation: str
    technical_maturity: Optional[str] = None
    references: List[Reference] = Field(default_factory=list)


class MarketAssessment(BaseModel):
    market_size: str
    growth_drivers: List[str] = Field(default_factory=list)
    target_industries: List[str] = Field(default_factory=list)
    scalability: str
    commercialization_risk: Optional[str] = None
    references: List[Reference] = Field(default_factory=list)


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


class FinalReport(BaseModel):
    html: str
