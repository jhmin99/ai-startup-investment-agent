from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """Vector search 결과 1개."""

    content: str
    score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompanyOverview(BaseModel):
    representative: Optional[str] = None
    founded_year: Optional[str] = None
    region: Optional[str] = None
    industry: Optional[str] = None
    main_product: Optional[str] = None
    venture_type: Optional[str] = None
    investment_status: Optional[str] = None
    website: Optional[str] = None

class TechnologySection(BaseModel):
    summary: Optional[str] = None
    maturity: Optional[str] = None
    patent_ip: Optional[str] = None


class StrengthsAndLimitations(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class DifferentiationSection(BaseModel):
    description: Optional[str] = None
    core_advantages: List[str] = Field(default_factory=list)


class PerformanceSection(BaseModel):
    customers_references: Optional[str] = None
    sales_growth: Optional[str] = None
    adoption_cases: Optional[str] = None


class StartupProfile(BaseModel):
    company_name: str
    score: float = Field(ge=0.0, le=1.0)
    source_file: Optional[str] = None
    pages: List[int] = Field(default_factory=list)

    company_overview: CompanyOverview = Field(default_factory=CompanyOverview)
    technology: TechnologySection = Field(default_factory=TechnologySection)
    strengths_and_limitations: StrengthsAndLimitations = Field(default_factory=StrengthsAndLimitations)
    differentiation: DifferentiationSection = Field(default_factory=DifferentiationSection)
    performance: PerformanceSection = Field(default_factory=PerformanceSection)

    references: List[str] = Field(default_factory=list)

    # 원문 보존 (파싱 실패해도 항상 유지)
    raw_texts: List[str] = Field(default_factory=list)
    merged_raw_text: str = ""
    metadata_list: List[Dict[str, Any]] = Field(default_factory=list)


class StartupSearchOutput(BaseModel):
    """startup_search 에이전트 출력."""

    user_query: str
    normalized_query: str
    retrieved_docs: List[RetrievedDocument]
    startup_profiles: List[StartupProfile]
    search_confidence: float = Field(ge=0.0, le=1.0)
    need_query_refinement: bool
