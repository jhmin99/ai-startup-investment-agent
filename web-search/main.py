"""
웹 검색 에이전트 CLI

사용법:
    python main.py "스타트업이름"
    python main.py "스타트업1" "스타트업2" "스타트업3"
    python main.py "스타트업이름" --output result.json
"""
import argparse
from typing import Dict, Any, List

from agent import WebSearchAgent


# ============================================================
# CLI 설정
# ============================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="스타트업 투자 분석을 위한 웹 검색 에이전트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py "수퍼빈"
    python main.py "수퍼빈" "뉴빌리티" "업스테이지"
    python main.py "수퍼빈" --output result.json
        """
    )
    parser.add_argument(
        "startup_names",
        type=str,
        nargs='+',
        help="분석할 스타트업 이름 (여러 개 가능)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="사용할 LLM 모델 (기본값: gpt-4o)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (선택)"
    )
    
    return parser.parse_args()


# ============================================================
# 결과 출력
# ============================================================

def print_result(result: Dict[str, Any]) -> None:
    """단일 스타트업 검색 결과 출력"""
    startup_name = result.get("startup_name", "")
    print(f"\n📊 {startup_name} 검색 결과")
    print("-" * 40)
    
    categories = [
        ("📈 시장성", "market"),
        ("🔬 기술력", "technology"),
        ("⚔️ 경쟁력", "competition"),
        ("📉 실적", "performance"),
    ]
    
    for label, key in categories:
        findings = result.get(key, [])
        print(f"\n{label}")
        if findings:
            for f in findings:
                print(f"  • {f}")
        else:
            print("  • 검색 결과 없음")


def print_results(results: List[Dict[str, Any]]) -> None:
    """여러 스타트업 검색 결과 출력"""
    for result in results:
        print_result(result)
        print("\n" + "=" * 50)


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """검색 결과를 JSON 파일로 저장"""
    import json
    
    # 단일 결과면 리스트 벗기기
    output = results[0] if len(results) == 1 else results
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ 결과가 '{output_path}'에 저장되었습니다.")


# ============================================================
# 메인 실행
# ============================================================

def main() -> List[Dict[str, Any]]:
    """메인 실행 함수"""
    args = parse_args()
    
    startup_names = args.startup_names
    print(f"\n🔍 {len(startup_names)}개 스타트업 검색 시작: {', '.join(startup_names)}")
    print("=" * 50)
    
    # 에이전트 생성
    agent = WebSearchAgent(model=args.model)
    
    # 각 스타트업 검색
    results = []
    for name in startup_names:
        print(f"\n⏳ '{name}' 검색 중...")
        result = agent.run(name)
        results.append(result)
    
    # 결과 출력
    print_results(results)
    
    # 파일 저장
    if args.output:
        save_results(results, args.output)
    
    return results


if __name__ == "__main__":
    main()
