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


class StartupRetriever:
    """
    pgvector 기반 similarity search 담당.

    - 기본 테이블: public.rag_vector_store_startup_search (config.py에서 변경 가능)
    - 검색 연산자: cosine distance (<=>) + vector_cosine_ops 인덱스
    """

    def __init__(self, conn=None, schema_name: Optional[str] = None, table_name: Optional[str] = None):
        settings = get_settings()
        self.schema_name = schema_name or settings["vectorstore"]["schema_name"]
        self.table_name = table_name or settings["vectorstore"]["startup_search_table"]
        self.conn = conn or get_psycopg_connection()

    def search(self, query: str, k: int = 5) -> List[RetrievedDocument]:
        """
        query를 동일 임베딩 모델로 임베딩하고 상위 k개 문서를 반환.
        score는 0~1로 정규화되며 높을수록 관련성이 높음.
        """
        if k <= 0:
            return []
        q = (query or "").strip()
        if not q:
            return []

        qvec: np.ndarray = embed_query(q)  # float32 numpy

        stmt = sql.SQL(
            """
            SELECT
              content,
              metadata,
              (embedding <=> %s) AS distance
            FROM {}.{}
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(sql.Identifier(self.schema_name), sql.Identifier(self.table_name))

        with self.conn.cursor() as cur:
            rows = cur.execute(stmt, (qvec, qvec, k)).fetchall()

        docs: List[RetrievedDocument] = []
        for r in rows:
            distance = r.get("distance", 2.0)
            docs.append(
                RetrievedDocument(
                    content=r.get("content") or "",
                    metadata=r.get("metadata") or {},
                    score=_distance_to_score(distance),
                )
            )

        return docs
