from __future__ import annotations

import argparse
from typing import Any, Dict

from .graph import SupervisorConfig, build_supervisor_graph


def run_supervisor(user_query: str, *, config: SupervisorConfig | None = None) -> Dict[str, Any]:
    """
    Supervisor 단독 실행 진입점.

    - 테스트/서버는 분리
    - LangGraph state를 그대로 반환 (후속에서 state로 연결 가능)
    """
    graph = build_supervisor_graph(config)
    state = {"user_query": user_query}
    out = graph.invoke(state)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph supervisor runner")
    p.add_argument("user_query", type=str, help="사용자 질의")
    p.add_argument("--k", type=int, default=10, help="startup_search k (default: 10)")
    p.add_argument("--max-startups", type=int, default=3, help="downstream 후보 수 (default: 3)")
    p.add_argument("--no-tech-summary", action="store_true", help="기술요약 노드 비활성화")
    p.add_argument("--no-web-search", action="store_true", help="웹검색 노드 비활성화")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = SupervisorConfig(
        startup_search_k=args.k,
        max_startups_for_downstream=args.max_startups,
        enable_technical_summary=not args.no_tech_summary,
        enable_web_search=not args.no_web_search,
    )
    out = run_supervisor(args.user_query, config=cfg)
    # 결과 출력은 호출 측에서 하도록 하고, CLI에서는 최소만 출력
    final_report = out.get("final_report")
    if isinstance(final_report, str) and final_report.strip():
        print(final_report)
    else:
        print(out)


if __name__ == "__main__":
    main()

