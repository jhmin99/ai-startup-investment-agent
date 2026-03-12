"""
질문 재정의 에이전트 CLI

사용법:
    python -m query_refinement.main "요즘 뜨는 로봇 회사 뭐야?" --result-count 0
    python query_refinement/main.py "국내 물류 로봇 스타트업 찾아줘" --top-score 0.18 --top-score 0.14
    python -m query_refinement.main "비전 검사 같은 거 하는 데" --result-count 0 --output result.json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

try:
    from query_refinement.agent import QueryRefinementAgent
    from query_refinement.state import QueryRefinementState
except ImportError:
    from agent import QueryRefinementAgent
    from state import QueryRefinementState


# ============================================================
# CLI 설정
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""

    parser = argparse.ArgumentParser(
        description="Startup Search fallback용 질문 재정의 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python -m query_refinement.main "요즘 뜨는 로봇 회사 뭐야?" --result-count 0
    python query_refinement/main.py "국내 물류 로봇 스타트업 찾아줘" --top-score 0.18 --top-score 0.14
    python -m query_refinement.main "비전 검사 같은 거 하는 데" --result-count 0 --output result.json
        """,
    )
    parser.add_argument(
        "question",
        type=str,
        help="재정의할 사용자 질문",
    )
    parser.add_argument(
        "--result-count",
        type=int,
        default=None,
        help="1차 검색 결과 개수",
    )
    parser.add_argument(
        "--top-score",
        action="append",
        type=float,
        default=[],
        help="1차 검색 top similarity score, 여러 번 줄 수 있다",
    )
    parser.add_argument(
        "--failure-reason",
        type=str,
        default="",
        help="디버깅용 검색 실패 메모",
    )
    parser.add_argument(
        "--noisy-results",
        action="store_true",
        help="1차 검색 결과가 잡음이 많았는지 표시",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로",
    )
    return parser.parse_args()


# ============================================================
# 상태 생성
# ============================================================


def build_state(args: argparse.Namespace) -> QueryRefinementState:
    """CLI 입력을 질문 재정의 상태로 변환한다."""

    state: QueryRefinementState = {"raw_question": args.question.strip()}

    if (
        args.result_count is not None
        or args.top_score
        or args.failure_reason
        or args.noisy_results
    ):
        state["retrieval_feedback"] = {
            "result_count": args.result_count,
            "top_scores": args.top_score,
            "failure_reason": args.failure_reason,
            "noisy_results": args.noisy_results,
        }

    return state


# ============================================================
# 결과 출력
# ============================================================


def print_result(result: Dict[str, Any]) -> None:
    """질문 재정의 결과를 화면에 출력한다."""

    print("\n질문 재정의 결과")
    print("-" * 40)
    print(f"refined_query: {result.get('refined_query', '')}")
    print("reformulated_queries:")
    for query in result.get("reformulated_queries", []):
        print(f"  - {query}")
    print(f"retry_strategy: {result.get('retry_strategy', '')}")


def save_result(result: Dict[str, Any], output_path: str) -> None:
    """질문 재정의 결과를 JSON 파일로 저장한다."""

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"\n결과가 '{output_path}'에 저장되었습니다.")


# ============================================================
# 메인 실행
# ============================================================


def main() -> Dict[str, Any]:
    """메인 실행 함수."""

    args = parse_args()
    state = build_state(args)

    agent = QueryRefinementAgent()
    result = agent.refine(state)

    print_result(result)

    if args.output:
        save_result(result, args.output)

    return result


if __name__ == "__main__":
    main()
