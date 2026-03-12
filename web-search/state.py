"""
웹 검색 에이전트 상태 정의

SearchResult: 개별 검색 결과
CategoryFindings: 카테고리별 정제된 정보
WebSearchState: 에이전트 전체 상태 (LangGraph용)
"""
from typing import TypedDict, Annotated, List, Optional, Dict
from langgraph.graph.message import add_messages


# ============================================================
# 데이터 모델
# ============================================================

class SearchResult(TypedDict):
    """개별 검색 결과 (원본)"""
    title: str       # 검색 결과 제목
    url: str         # 출처 URL
    content: str     # 본문 내용
    category: str    # 카테고리 (market/technology/competition/performance)


class CategoryFindings(TypedDict):
    """카테고리별 정제된 정보 (투자 판단 에이전트용)"""
    findings: List[str]      # 핵심 팩트 리스트


# ============================================================
# 에이전트 출력 (투자 판단 에이전트 전달용)
# ============================================================

class WebSearchOutput(TypedDict):
    """
    웹 검색 에이전트 출력 형식
    
    투자 판단 에이전트가 점수 산출에 활용할 수 있는 구조화된 형태.
    """
    startup_name: str
    
    # 카테고리별 핵심 팩트 리스트
    market: List[str]
    technology: List[str]
    competition: List[str]
    performance: List[str]


# ============================================================
# 에이전트 상태 (LangGraph StateGraph용)
# ============================================================

class WebSearchState(TypedDict):
    """
    웹 검색 에이전트 상태
    
    상위 그래프와 통합 시 필요한 필드만 사용 가능.
    """
    # 입력
    startup_name: str
    
    # 메시지 히스토리
    messages: Annotated[list, add_messages]
    
    # 에러
    error: Optional[str]
