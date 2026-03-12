from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class SupervisorState(TypedDict, total=False):
    """
    LangGraph 상위 Supervisor 상태.

    원칙:
    - 각 노드는 필요한 입력만 읽고, 자신의 결과를 state에 추가한다.
    - 원문/근거 보존: startup_search 결과(StartupSearchOutput)는 그대로 state에 보관한다.
    """

    # 입력
    user_query: str

    # 질문 정제 결과
    refined_query: str
    reformulated_queries: List[str]
    retry_strategy: str

    # startup_search 결과(전체를 그대로)
    startup_search: Dict[str, Any]  # StartupSearchOutput.model_dump()

    # 선택된 후보(후속 에이전트 입력용)
    selected_startup_names: List[str]

    # 기술 요약 결과(여러 회사 가능)
    technical_summaries: List[Dict[str, Any]]

    # 웹 검색 결과(여러 회사 가능)
    web_search_results: List[Dict[str, Any]]

    # 최종 보고서(LLM 없이도 생성 가능하도록 텍스트로)
    final_report: str

    # 에러/메모
    error: Optional[str]

