"""
질문 재정의 에이전트 상태 정의

RetrievalFeedback: Startup Search 1차 검색 상태
QueryRefinementState: 질문 재정의 에이전트 입력 상태
QueryRefinementOutput: 재검색용 질의 출력 상태
"""

from typing import List, Optional, TypedDict


# ============================================================
# 입력 상태
# ============================================================


class RetrievalFeedback(TypedDict, total=False):
    """1차 검색에서 관측된 상태"""

    result_count: Optional[int]
    top_scores: List[float]
    noisy_results: bool
    failure_reason: str


class QueryRefinementState(TypedDict, total=False):
    """질문 재정의 에이전트 입력 상태"""

    raw_question: str
    retrieval_feedback: RetrievalFeedback


# ============================================================
# 출력 상태
# ============================================================


class QueryRefinementOutput(TypedDict):
    """질문 재정의 결과"""

    refined_query: str
    reformulated_queries: List[str]
    retry_strategy: str


# ============================================================
# 검증 함수
# ============================================================


def validate_query_refinement_state(state: QueryRefinementState) -> None:
    """필수 입력만 최소 검증한다."""

    raw_question = state.get("raw_question", "")
    if not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError("raw_question must not be empty.")

    feedback = state.get("retrieval_feedback")
    if feedback is None:
        return

    result_count = feedback.get("result_count")
    if result_count is not None and result_count < 0:
        raise ValueError("result_count must be greater than or equal to 0.")
