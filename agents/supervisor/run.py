from __future__ import annotations

import argparse
from html import escape
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


def _wrap_text_as_html(report_text: str, title: str = "로봇 스타트업 투자 보고서") -> str:
    safe = escape(report_text or "")
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
</head>
<body>
  <pre style="white-space: pre-wrap; font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;">
{safe}
  </pre>
</body>
</html>
"""


def _looks_like_html(s: str) -> bool:
    t = (s or "").lstrip().lower()
    return t.startswith("<!doctype html") or t.startswith("<html") or "<body" in t


def _save_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content or "")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LangGraph supervisor runner")
    p.add_argument("user_query", type=str, help="사용자 질의")
    p.add_argument("--k", type=int, default=10, help="startup_search k (default: 10)")
    p.add_argument("--max-startups", type=int, default=3, help="downstream 후보 수 (default: 3)")
    p.add_argument("--no-tech-summary", action="store_true", help="기술요약 노드 비활성화")
    p.add_argument("--no-web-search", action="store_true", help="웹검색 노드 비활성화")
    p.add_argument("--output-html", type=str, default=None, help="final_report를 HTML 파일로 저장")
    p.add_argument("--output-pdf", type=str, default=None, help="final_report를 PDF 파일로 저장(weasyprint 필요)")
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
    final_report = out.get("final_report")

    # 1) 파일 저장 옵션 처리
    if isinstance(final_report, str) and final_report.strip():
        if args.output_html:
            html_out = final_report if _looks_like_html(final_report) else _wrap_text_as_html(final_report)
            _save_text(args.output_html, html_out)
            print(f"saved html: {args.output_html}")

        if args.output_pdf:
            # HTML → PDF 변환은 final_report_agent 유틸을 재사용
            from agents.final_report_agent.utils import html_to_pdf

            html = final_report if _looks_like_html(final_report) else _wrap_text_as_html(final_report)
            try:
                html_to_pdf(html, args.output_pdf)
                print(f"saved pdf: {args.output_pdf}")
            except Exception as exc:
                # PDF 변환은 시스템 의존성이 필요할 수 있어, 실패해도 전체 실행은 계속한다.
                print(f"failed to save pdf: {args.output_pdf}")
                print(f"reason: {exc}")

    # 2) CLI 기본 출력
    if isinstance(final_report, str) and final_report.strip():
        print(final_report)
    else:
        print(out)


if __name__ == "__main__":
    main()

