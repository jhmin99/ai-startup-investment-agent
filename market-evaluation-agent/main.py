"""
시장성 평가 에이전트 CLI

사용법:
    python main.py sample_input.json
    python main.py sample_input.json --output result.json
    python /Volumes/T7/AI-STARTUP-INVESTMENT-AGENT/ai-startup-investment-agent/market-evaluation-agent/main.py /Volumes/T7/AI-STARTUP-INVESTMENT-AGENT/ai-startup-investment-agent/market-evaluation-agent/sample_input.json --output result.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from agent import MarketEvaluationAgent


# ============================================================
# CLI 설정
# ============================================================


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""

    parser = argparse.ArgumentParser(
        description="시장성 평가 에이전트 - market_eval vector store 기반 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py sample_input.json
    python main.py sample_input.json --output result.json
        """,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="입력 JSON 파일 경로",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="사용할 LLM 모델",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로",
    )
    return parser.parse_args()


# ============================================================
# 입출력 처리
# ============================================================


def load_input(file_path: str) -> List[Dict[str, Any]]:
    """입력 JSON 파일을 읽는다."""

    input_path = Path(file_path).expanduser()
    if not input_path.exists():
        sample_path = Path(__file__).with_name("sample_input.json")
        raise FileNotFoundError(
            "입력 JSON 파일을 찾을 수 없습니다: "
            f"'{file_path}'\n"
            f"- 현재 작업 디렉터리: '{Path.cwd()}'\n"
            f"- 예시 입력 파일: '{sample_path}'\n"
            "- 이 에이전트는 `rag_vector_store_market_eval`에 적재된 시장 보고서 RAG를 조회합니다.\n"
            "- 실행 예시: "
            "python market-evaluation-agent/main.py "
            "market-evaluation-agent/sample_input.json --output result.json"
        )

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return payload

    raise ValueError("입력 JSON은 단일 객체 또는 객체 리스트여야 합니다.")


def print_result(result: Dict[str, Any]) -> None:
    """시장성 평가 결과를 출력한다."""

    print("\n시장성 평가 결과")
    print("-" * 40)
    print(f"startup_name: {result.get('startup_name', '')}")
    print(f"market_query: {result.get('market_query', '')}")
    print(f"search_query: {result.get('search_query', '')}")
    print(f"\nmarket_summary: {result.get('market_summary', '')}")
    print(f"market_size: {result.get('market_size', '')}")
    print(f"competition_analysis: {result.get('competition_analysis', '')}")
    print(f"customer_adoption: {result.get('customer_adoption', '')}")
    print("\ngrowth_drivers:")
    for item in result.get("growth_drivers", []):
        print(f"  - {item}")
    print("\nkey_risks:")
    for item in result.get("key_risks", []):
        print(f"  - {item}")
    print("\nevidence:")
    for item in result.get("evidence", []):
        print(f"  - {item}")


def save_result(result: Dict[str, Any], output_path: str) -> None:
    """결과를 JSON 파일로 저장한다."""

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"\n결과가 '{output_path}'에 저장되었습니다.")


def print_results(results: List[Dict[str, Any]]) -> None:
    """여러 시장성 평가 결과를 출력한다."""

    for result in results:
        print_result(result)
        print("\n" + "=" * 50)


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """단일/복수 결과를 JSON 파일로 저장한다."""

    output: Any = results[0] if len(results) == 1 else results

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(f"\n결과가 '{output_path}'에 저장되었습니다.")


# ============================================================
# 메인 실행
# ============================================================


def main() -> List[Dict[str, Any]]:
    """메인 실행 함수."""

    args = parse_args()
    try:
        payloads = load_input(args.input_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    agent = MarketEvaluationAgent(model=args.model)
    results: List[Dict[str, Any]] = []
    try:
        for payload in payloads:
            results.append(agent(payload))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print_results(results)

    if args.output:
        save_results(results, args.output)

    return results


if __name__ == "__main__":
    main()
