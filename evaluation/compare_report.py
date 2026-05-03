"""
Before / After RAGAS 수치 비교 리포트.

사용 순서:
  1. .venv/bin/python evaluation/ragas_before.py   → evaluation/before_scores.csv 생성
  2. .venv/bin/python evaluation/ragas_after.py    → evaluation/after_scores.csv 생성
  3. .venv/bin/python evaluation/compare_report.py → 비교 표 출력

필요 패키지:
  .venv/bin/pip install ragas datasets pandas
"""
from __future__ import annotations

import os
import sys

BEFORE_CSV = os.path.join(os.path.dirname(__file__), "before_scores.csv")
AFTER_CSV  = os.path.join(os.path.dirname(__file__), "after_scores.csv")


def load_scores(path: str):
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] {path} 읽기 실패: {e}")
        return None


def print_table(before, after, metrics: list[str]) -> None:
    print("\n" + "=" * 62)
    print("  RAG 품질 개선 결과 (RAGAS Before → After)")
    print("=" * 62)
    print(f"  {'지표':<28} {'Before':>8} {'After':>8} {'개선':>8}")
    print("-" * 62)
    for m in metrics:
        if m not in before.columns or m not in after.columns:
            continue
        b = before[m].mean()
        a = after[m].mean()
        arrow = "▲" if a > b else ("▼" if a < b else "─")
        print(f"  {m:<28} {b:>8.3f} {a:>8.3f} {arrow}{abs(a-b):>7.3f}")
    print("=" * 62)

    # 회사별 상세
    if "company" in before.columns and "company" in after.columns:
        print("\n  [회사별 context_precision 비교]")
        print(f"  {'회사':<16} {'Before':>8} {'After':>8} {'개선':>8}")
        print("-" * 48)
        companies = before["company"].unique()
        for company in companies:
            b_sub = before[before["company"] == company]
            a_sub = after[after["company"] == company]
            if b_sub.empty or a_sub.empty:
                continue
            b_cp = b_sub["context_precision"].mean()
            a_cp = a_sub["context_precision"].mean()
            arrow = "▲" if a_cp > b_cp else ("▼" if a_cp < b_cp else "─")
            print(f"  {company:<16} {b_cp:>8.3f} {a_cp:>8.3f} {arrow}{abs(a_cp-b_cp):>7.3f}")
        print()

        # 청크 중복률: before에서 5개 회사가 동일 chunk를 가져오는 비율
        _print_chunk_overlap(before, after)


def _print_chunk_overlap(before, after) -> None:
    """
    5개 회사 간 검색 결과 중복률.
    contexts 컬럼이 존재하면 회사 간 겹치는 문장 비율을 계산.
    """
    import ast

    def get_all_contexts(df) -> dict[str, set[str]]:
        result = {}
        for company in df["company"].unique():
            sub = df[df["company"] == company]
            contexts_set: set[str] = set()
            for raw in sub.get("contexts", []):
                try:
                    items = ast.literal_eval(str(raw)) if isinstance(raw, str) else raw
                    for item in (items or []):
                        contexts_set.add(str(item).strip()[:100])
                except Exception:
                    pass
            result[company] = contexts_set
        return result

    if "contexts" not in before.columns:
        return

    print("  [회사 간 청크 중복률]")
    before_ctx = get_all_contexts(before)
    after_ctx  = get_all_contexts(after)

    def overlap_rate(ctx_dict: dict[str, set[str]]) -> float:
        companies = list(ctx_dict.keys())
        if len(companies) < 2:
            return 0.0
        total, overlapping = 0, 0
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                a_set = ctx_dict[companies[i]]
                b_set = ctx_dict[companies[j]]
                union = a_set | b_set
                inter = a_set & b_set
                if union:
                    total += len(union)
                    overlapping += len(inter)
        return overlapping / total if total > 0 else 0.0

    b_rate = overlap_rate(before_ctx) * 100
    a_rate = overlap_rate(after_ctx) * 100
    arrow = "▼" if a_rate < b_rate else "▲"
    print(f"  {'청크 중복률 (5사 평균)':<28} {b_rate:>7.1f}% {a_rate:>7.1f}% {arrow}{abs(b_rate-a_rate):.1f}%p")
    print()


def main() -> None:
    if not os.path.exists(BEFORE_CSV):
        print(f"[ERROR] {BEFORE_CSV} 파일이 없습니다. 먼저 ragas_before.py를 실행하세요.")
        sys.exit(1)
    if not os.path.exists(AFTER_CSV):
        print(f"[ERROR] {AFTER_CSV} 파일이 없습니다. 먼저 ragas_after.py를 실행하세요.")
        sys.exit(1)

    before = load_scores(BEFORE_CSV)
    after  = load_scores(AFTER_CSV)
    if before is None or after is None:
        sys.exit(1)

    metrics = ["context_precision", "context_recall"]
    print_table(before, after, metrics)

    print("  ── 자소서 스토리 포인트 ──────────────────────────────")
    print("  - Before: 5개 기업 모두 동일 청크 반환 → context_precision 낮음")
    print("  - 개선: 메타데이터 필터링 + 쿼리 증강 + MMR 도입")
    print("  - After:  기업별 관련 청크만 반환 → precision/recall 향상")
    print("  ──────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
