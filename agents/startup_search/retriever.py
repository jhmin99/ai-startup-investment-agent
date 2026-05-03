from __future__ import annotations

from typing import List, Optional

import numpy as np
from psycopg import sql

from config import get_settings

from .schemas import RetrievedDocument
from .utils import embed_query, get_psycopg_connection


def _distance_to_score(distance: float) -> float:
    """
    pgvector cosine distance(<=>)를 사람이 이해하기 쉬운 score로 변환.

    - cosine distance range: [0, 2]
    - score: 1 - distance/2  (0~1로 clamp)
    """
    s = 1.0 - (float(distance) / 2.0)
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도 (0~1)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _mmr_select(
    query_vec: np.ndarray,
    candidates: list[tuple[RetrievedDocument, np.ndarray]],
    k: int,
    lambda_: float = 0.5,
) -> list[RetrievedDocument]:
    """
    Maximal Marginal Relevance 선택.

    - relevance: 쿼리와 유사도 높을수록 좋음
    - redundancy: 이미 선택된 것과 유사도 낮을수록 좋음
    - score = lambda * relevance - (1 - lambda) * max_redundancy

    lambda_=1.0 이면 순수 유사도 순(MMR 비활성), 0.5가 균형점.
    """
    if not candidates:
        return []

    selected: list[tuple[RetrievedDocument, np.ndarray]] = []
    remaining = list(candidates)

    while len(selected) < k and remaining:
        scores = []
        for doc, vec in remaining:
            relevance = _cosine_sim(query_vec, vec)
            if selected:
                redundancy = max(_cosine_sim(vec, sv) for _, sv in selected)
            else:
                redundancy = 0.0
            scores.append(lambda_ * relevance - (1 - lambda_) * redundancy)

        best_idx = int(np.argmax(scores))
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)

    return [doc for doc, _ in selected]


class StartupRetriever:
    """
    pgvector 기반 similarity search 담당.

    개선 사항:
    - company_name 파라미터로 메타데이터 필터링 지원 (회사별 청크만 검색)
    - MMR(Maximal Marginal Relevance)로 중복 청크 제거
    - 기본 후보 수 candidate_k=20 → 최종 k개 선택
    """

    def __init__(self, conn=None, schema_name: Optional[str] = None, table_name: Optional[str] = None):
        settings = get_settings()
        self.schema_name = schema_name or settings["vectorstore"]["schema_name"]
        self.table_name = table_name or settings["vectorstore"]["startup_search_table"]
        self.conn = conn or get_psycopg_connection()

    def search(
        self,
        query: str,
        k: int = 10,
        company_name: Optional[str] = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        candidate_k: int = 20,
    ) -> List[RetrievedDocument]:
        """
        query를 임베딩하고 상위 k개 문서를 반환.

        Args:
            query: 검색 질의
            k: 최종 반환 문서 수 (기본 10)
            company_name: 지정 시 해당 회사 청크만 검색 (metadata->>'company_name' 필터)
            use_mmr: True이면 MMR로 다양성 확보, False이면 단순 유사도 순
            mmr_lambda: MMR 균형 계수 (1=순수 유사도, 0=순수 다양성, 기본 0.5)
            candidate_k: MMR 후보 수 (use_mmr=True일 때 내부적으로 이만큼 먼저 조회)
        """
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []

        qvec: np.ndarray = embed_query(q)

        fetch_k = candidate_k if use_mmr else k

        # ── SQL: company_name 필터 조건 분기 ──────────────────────
        if company_name:
            # LIKE 매칭: "(주)파워로보틱스", "파워로보틱스 주식회사" 등 변형에도 대응
            stmt = sql.SQL(
                """
                SELECT
                  content,
                  metadata,
                  embedding,
                  (embedding <=> %s) AS distance
                FROM {schema}.{table}
                WHERE metadata->>'company_name' LIKE %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """
            ).format(
                schema=sql.Identifier(self.schema_name),
                table=sql.Identifier(self.table_name),
            )
            params = (qvec, f"%{company_name}%", qvec, fetch_k)
        else:
            stmt = sql.SQL(
                """
                SELECT
                  content,
                  metadata,
                  embedding,
                  (embedding <=> %s) AS distance
                FROM {schema}.{table}
                ORDER BY embedding <=> %s
                LIMIT %s
                """
            ).format(
                schema=sql.Identifier(self.schema_name),
                table=sql.Identifier(self.table_name),
            )
            params = (qvec, qvec, fetch_k)

        with self.conn.cursor() as cur:
            rows = cur.execute(stmt, params).fetchall()

        if not rows:
            return []

        # ── MMR 적용 ──────────────────────────────────────────────
        if use_mmr:
            candidates: list[tuple[RetrievedDocument, np.ndarray]] = []
            for r in rows:
                distance = r.get("distance", 2.0)
                doc = RetrievedDocument(
                    content=r.get("content") or "",
                    metadata=r.get("metadata") or {},
                    score=_distance_to_score(distance),
                )
                raw_emb = r.get("embedding")
                if raw_emb is not None:
                    vec = np.array(raw_emb, dtype=np.float32)
                else:
                    vec = qvec  # fallback
                candidates.append((doc, vec))

            return _mmr_select(qvec, candidates, k, lambda_=mmr_lambda)
        else:
            docs: List[RetrievedDocument] = []
            for r in rows[:k]:
                distance = r.get("distance", 2.0)
                docs.append(
                    RetrievedDocument(
                        content=r.get("content") or "",
                        metadata=r.get("metadata") or {},
                        score=_distance_to_score(distance),
                    )
                )
            return docs

    def search_without_improvements(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """
        개선 전 동작 재현용 메서드 (RAGAS before 측정에 사용).
        - company_name 필터 없음
        - MMR 없음
        - k=5
        """
        return self.search(query, k=k, company_name=None, use_mmr=False)
