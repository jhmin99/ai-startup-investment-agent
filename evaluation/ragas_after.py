"""
RAGAS After 측정 스크립트
- 개선된 retriever(company_name 필터 + 쿼리 증강 + MMR)로 동일 데이터셋 재측정
- 결과는 evaluation/after_scores.csv 로 저장

실행:
    cd /Users/jihong/Desktop/ai-startup-investment-agent
    .venv/bin/python evaluation/ragas_after.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.startup_search.retriever import StartupRetriever
from agents.startup_search.utils import normalize_query


EVAL_CSV = os.path.join(os.path.dirname(__file__), "eval_dataset.csv")


def load_dataset() -> list[dict]:
    rows = []
    with open(EVAL_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


# DB에 저장된 실제 company_name 값과 매핑
# (ingestion 시 파싱된 이름이 "(주)파워로보틱스" 등 접두/접미어를 포함할 수 있음)
COMPANY_NAME_MAP = {
    "파워로보틱스": "파워로보틱스",
    "로보프린트": "로보프린트",
    "지오로봇": "지오로봇",
    "필드로": "필드로",
    "고레로보틱스": "고레로보틱스",
}


def retrieve_contexts_improved(question: str, company_name: str, k: int = 10) -> list[str]:
    """
    개선된 검색:
    - 쿼리 증강: "{회사명} {질의}" 형태
    - 메타데이터 필터: LIKE '%company_name%' (부분 일치)
    - MMR: use_mmr=True (중복 청크 제거)
    - k=10
    """
    retriever = StartupRetriever()
    # 매핑된 검색 키워드 사용 (없으면 그대로)
    search_company = COMPANY_NAME_MAP.get(company_name, company_name)
    augmented_query = f"{company_name} {normalize_query(question)}"
    docs = retriever.search(
        augmented_query,
        k=k,
        company_name=search_company,
        use_mmr=True,
        mmr_lambda=0.5,
        candidate_k=20,
    )
    return [d.content for d in docs if d.content]


def run_evaluation(output_csv: str = "evaluation/after_scores.csv") -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
    except ImportError:
        print("[ERROR] ragas 또는 datasets 패키지가 설치되어 있지 않습니다.")
        print("  .venv/bin/pip install ragas datasets")
        sys.exit(1)

    rows = load_dataset()
    print(f"[ragas_after] 총 {len(rows)}개 질문 평가 시작 (k=10, company_name 필터 + MMR)")

    eval_rows = []
    for i, row in enumerate(rows):
        company = row["company"]
        question = row["question"]
        ground_truth = row["ground_truth"]

        print(f"  [{i+1}/{len(rows)}] {company} | {question[:30]}...")
        contexts = retrieve_contexts_improved(question, company_name=company, k=10)

        answer = " ".join(contexts[:2]) if contexts else "검색 결과 없음"

        eval_rows.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth,
        })

    dataset = Dataset.from_list(eval_rows)
    result = evaluate(dataset, metrics=[context_precision, context_recall])

    print("\n===== AFTER (개선 후) =====")
    print(result)

    df = result.to_pandas()
    df["company"] = [r["company"] for r in rows]
    df["question"] = [r["question"] for r in rows]
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n[ragas_after] 결과 저장 완료: {output_csv}")

    print("\n===== 회사별 context_precision =====")
    for company in df["company"].unique():
        sub = df[df["company"] == company]
        cp = sub["context_precision"].mean()
        cr = sub["context_recall"].mean()
        print(f"  {company:<15} precision={cp:.3f}  recall={cr:.3f}")


if __name__ == "__main__":
    run_evaluation()
