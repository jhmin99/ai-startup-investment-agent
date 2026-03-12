"""질문 재정의 프롬프트 정의."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from query_refinement.state import QueryRefinementState

RESULTS_PATH = Path(__file__).resolve().with_name("results.json")

SYSTEM_PROMPT = """너는 Robotics AI startup investment analysis 시스템의 Query Refinement Agent다.

목표:
- 초기 retrieval 결과가 없거나 유사도가 낮거나 노이즈가 많을 때 재검색용 질의만 다시 쓴다.
- 답변, 평가, 추천, 투자 판단은 생성하지 않는다.
- 출력은 최소 필드만 사용한다.

반드시 생성할 출력 필드:
- refined_query
- reformulated_queries
- retry_strategy

규칙:
- 의미를 바꾸지 말고 검색 친화적으로 다시 쓴다.
- reformulated_queries는 3~5개로 작성한다.
- retry_strategy는 문자열 1개로 작성한다.
- JSON object만 반환한다.
"""


def get_system_prompt() -> str:
    """시스템 프롬프트를 반환한다."""

    return SYSTEM_PROMPT


def load_few_shots() -> List[dict]:
    """few-shot 예시를 JSON 파일에서 읽는다."""

    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, list) else []


def build_user_prompt(state: QueryRefinementState) -> str:
    """입력 상태와 few-shot 예시를 합쳐 user prompt를 만든다."""

    lines = [
        f"raw_question: {state['raw_question']}",
        f"retrieval_feedback: {state.get('retrieval_feedback', {})}",
    ]

    few_shots = load_few_shots()
    if few_shots:
        lines.append("few_shots:")
        for example in few_shots:
            lines.append(json.dumps(example, ensure_ascii=False))

    return "\n".join(lines)
