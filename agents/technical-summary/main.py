"""
기술 요약 에이전트 CLI

사용법:
    python main.py input.json
    python main.py input.json --output result.json
    
입력 JSON 형식:
    단일 기업:
    {
        "startup_name": "수퍼빈",
        "technology_info": { ... }
    }
    
    여러 기업:
    [
        {"startup_name": "수퍼빈", "technology_info": { ... }},
        {"startup_name": "뉴빌리티", "technology_info": { ... }}
    ]
"""
import argparse
import json
from typing import Dict, Any, List

from agent import TechSummaryAgent


# ============================================================
# CLI 설정
# ============================================================

def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="기술 요약 에이전트 - 스타트업 기술 정보 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python main.py input.json
    python main.py input.json --output result.json
    python main.py input.json --model gpt-4o-mini
        """
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="입력 JSON 파일 경로"
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
# 입력 처리
# ============================================================

def load_input(file_path: str) -> List[Dict[str, Any]]:
    """입력 파일 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 단일 객체면 리스트로 변환
    if isinstance(data, dict):
        return [data]
    return data


# ============================================================
# 결과 출력
# ============================================================

def print_result(result: Dict[str, Any]) -> None:
    """단일 기업 결과 출력"""
    startup_name = result.get("startup_name", "")
    
    print(f"\n🔬 {startup_name} 기술 요약")
    print("-" * 40)
    
    print(f"\n💡 핵심 기술: {result.get('core_technology', '정보 없음')}")
    print(f"\n📝 기술 요약: {result.get('tech_summary', '정보 없음')}")
    
    print("\n✅ 기술 강점:")
    for s in result.get("tech_strengths", []):
        print(f"  • {s}")
    
    print("\n⚠️ 기술 약점/리스크:")
    for w in result.get("tech_weaknesses", []):
        print(f"  • {w}")
    
    print(f"\n🎯 차별화 포인트: {result.get('tech_differentiation', '정보 없음')}")
    
    patent = result.get("patent_count")
    rd_size = result.get("rd_team_size")
    if patent or rd_size:
        print("\n📊 정량 지표:")
        if patent:
            print(f"  • 특허 수: {patent}건")
        if rd_size:
            print(f"  • R&D 규모: {rd_size}")


def print_results(results: List[Dict[str, Any]]) -> None:
    """여러 기업 결과 출력"""
    for result in results:
        print_result(result)
        print("\n" + "=" * 50)


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """결과를 JSON 파일로 저장"""
    # messages 필드 제거 (출력용)
    clean_results = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != "messages"}
        clean_results.append(clean)
    
    # 단일 결과면 리스트 벗기기
    output = clean_results[0] if len(clean_results) == 1 else clean_results
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 결과가 '{output_path}'에 저장되었습니다.")


# ============================================================
# 메인 실행
# ============================================================

def main() -> List[Dict[str, Any]]:
    """메인 실행 함수"""
    args = parse_args()
    
    # 입력 파일 로드
    inputs = load_input(args.input_file)
    print(f"\n🔍 {len(inputs)}개 스타트업 기술 분석 시작")
    print("=" * 50)
    
    # 에이전트 생성
    agent = TechSummaryAgent(model=args.model)
    
    # 각 스타트업 분석
    results = []
    for inp in inputs:
        startup_name = inp.get("startup_name", "Unknown")
        print(f"\n⏳ '{startup_name}' 분석 중...")
        result = agent(inp)
        results.append(result)
    
    # 결과 출력
    print_results(results)
    
    # 파일 저장
    if args.output:
        save_results(results, args.output)
    
    return results


if __name__ == "__main__":
    main()
