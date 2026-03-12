"""
시장성 평가 에이전트 상태 정의

MarketEvaluationInput: 시장성 평가 입력 상태
MarketEvaluationOutput: 투자 판단 단계로 넘길 시장성 평가 결과
"""

from typing import List, TypedDict


# ============================================================
# 입력 상태
# ============================================================


class MarketEvaluationInput(TypedDict, total=False):
    """시장성 평가 에이전트 입력."""

    startup_name: str
    market_query: str
    company_context: str
    top_k: int


# ============================================================
# 출력 상태
# ============================================================


class MarketEvaluationOutput(TypedDict):
    """투자 판단 단계로 넘길 시장성 평가 결과."""

    startup_name: str
    market_query: str
    search_query: str
    market_summary: str
    market_size: str
    growth_drivers: List[str]
    competition_analysis: str
    customer_adoption: str
    key_risks: List[str]
    evidence: List[str]
