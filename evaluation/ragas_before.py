"""
RAGAS Before 측정 스크립트
- 현재(개선 전) retriever를 그대로 사용해 context_precision / context_recall을 측정
- 결과는 evaluation/before_scores.csv 로 저장

실행:
    cd /Users/jihong/Desktop/ai-startup-investment-agent
    .venv/bin/python evaluation/ragas_before.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.startup_search.retriever import StartupRetriever
from agents.startup_search.utils import normalize_query


# ── 데이터셋 로드 ───────────────────────────────────────────────
EVAL_CSV = os.path.join(os.path.dirname(__file__), "eval_dataset.csv")

def load_dataset() -> list[dict]:
    rows = []
    with open(EVAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# ── 검색 실행 (개선 전: company_name 필터 없음, k=5) ────────────
def retrieve_contexts(question: str, k: int = 5) -> list[str]:
    retriever = StartupRetriever()
    docs = retriever.search(normalize_query(question), k=k)
    return [d.content for d in docs if d.content]


# ── RAGAS 평가 ────────────────────────────────────────────────
def run_evaluation(output_csv: str = "evaluation/before_scores.csv") -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
    except ImportError:
        print("[ERROR] ragas 또는 datasets 패키지가 설치되어 있지 않습니다.")
        print("  .venv/bin/pip install ragas datasets")
        sys.exit(1)

    rows = load_dataset()
    print(f"[ragas_before] 총 {len(rows)}개 질문 평가 시작 (k=5, 필터 없음)")

    eval_rows = []
    for i, row in enumerate(rows):
        company = row["company"]
        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"  [{i+1}/{len(rows)}] {company} | {question[:30]}...")
        contexts = retrieve_contexts(question, k=5)

        # 검색된 컨텍스트를 바탕으로 answer 생성 (단순 연결)
        answer = " ".join(contexts[:2]) if contexts else "검색 결과 없음"

        eval_rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(eval_rows)
    result = evaluate(dataset, metrics=[context_precision, context_recall])

    print("\n===== BEFORE (개선 전) =====")
    print(result)

    # 상세 점수 저장
    df = result.to_pandas()
    df["company"] = [r["company"] for r in rows]
    df["question"] = [r["question"] for r in rows]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[ragas_before] 결과 저장 완료: {output_csv}")

    # 회사별 평균도 출력
    print("\n===== 회사별 context_precision =====")
    for company in df["company"].unique():
        sub = df[df["company"] == company]
        cp = sub["context_precision"].mean()
        cr = sub["context_recall"].mean()
        print(f"  {company:<15} precision={cp:.3f}  recall={cr:.3f}")


if __name__ == "__main__":
    run_evaluation()
