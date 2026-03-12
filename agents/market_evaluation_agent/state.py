"""
시장성 평가 에이전트 상태 정의 (agents/market_evaluation_agent).
"""

from typing import List, TypedDict


class MarketEvaluationInput(TypedDict, total=False):
    """시장성 평가 에이전트 입력."""

    startup_name: str
    market_query: str
    company_context: str
    top_k: int


class MarketEvaluationOutput(TypedDict):
    """투자 판단 단계로 넘길 시장성 평가 결과."""

    startup_name: str
    market_query: str
    search_query: str
    market_summary: str
    market_size: str
    growth_drivers: List[str]
    target_industries: List[str]
    competition_analysis: str
    customer_adoption: str
    key_risks: List[str]
    evidence: List[str]

