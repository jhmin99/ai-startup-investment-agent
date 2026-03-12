from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import numpy as np
import psycopg
from langchain_community.embeddings import HuggingFaceEmbeddings
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config import get_settings
from vectorstore import EMBEDDING_MODEL

from .schemas import RetrievedDocument, StartupProfile


def normalize_query(query: str) -> str:
    """질의 정규화: 앞뒤 공백 제거 + 다중 공백 축약."""
    q = (query or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def deduplicate_profiles(profiles: List[StartupProfile]) -> List[StartupProfile]:
    """company_name 기준 dedup. 점수가 더 높은 프로필을 남김."""
    best = {}
    for p in profiles:
        key = (p.company_name or "unknown").strip().lower()
        if key not in best or p.score > best[key].score:
            best[key] = p
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def calculate_search_confidence(retrieved_docs: List[RetrievedDocument], candidate_count: int) -> float:
    """
    규칙 기반 confidence.
    - 문서가 없으면 0
    - 상위 3개 평균 점수 기반
    - 후보가 너무 적으면 소폭 하향
    """
    if not retrieved_docs:
        return 0.0
    top = retrieved_docs[:3]
    avg = float(sum(d.score for d in top) / len(top))
    # 후보 수가 0이면 크게 하향, 1이면 소폭 하향
    if candidate_count == 0:
        avg *= 0.4
    elif candidate_count == 1:
        avg *= 0.85
    return float(min(1.0, max(0.0, avg)))


def should_refine_query(confidence: float, candidate_count: int) -> bool:
    """간단 규칙: confidence < 0.6 또는 후보 0이면 refine 필요."""
    return confidence < 0.6 or candidate_count == 0


def extract_company_name_from_text(text: str) -> str | None:
    """
    회사명 추출 휴리스틱(개선판).

    우선순위:
    1) "1. (주)파워로보틱스" 같은 라인
    2) "(주)파워로보틱스는 ..." 같은 문장
    """
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines()[:30] if l.strip()]
    head = "\n".join(lines)

    # 1) "1. 회사명"
    m = re.search(r"^\s*\d+\.\s*([^\n]{2,80})$", head, flags=re.MULTILINE)
    if m:
        return _normalize_company_name(m.group(1))

    # 2) "회사명은 ..." 패턴
    m = re.search(r"^\s*([^\s][^\n]{1,40}?)\s*는\s", head, flags=re.MULTILINE)
    if m:
        cand = _normalize_company_name(m.group(1))
        if cand:
            return cand
    return None


def _normalize_company_name(name: str) -> str | None:
    n = (name or "").strip()
    if not n:
        return None
    n = re.sub(r"^[\-\*\•\s]+", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    # 섹션 제목처럼 보이는 값 제외
    if n in {"회사 개요", "기술 요약", "기술적 강점 및 한계", "경쟁사 대비 차별성"}:
        return None
    # 너무 긴 경우 앞부분만
    if len(n) > 80:
        n = n[:80].strip()
    return n


@lru_cache
def get_embedding_model() -> HuggingFaceEmbeddings:
    """jhgan/ko-sbert-nli 임베딩 모델 (프로세스 내 캐시)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def embed_query(text: str) -> np.ndarray:
    """질의를 임베딩하고 numpy vector로 반환."""
    vec = get_embedding_model().embed_query(text)
    return np.array(vec, dtype=np.float32)


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """문서 chunk들을 임베딩하고 numpy vector 리스트로 반환."""
    vectors = get_embedding_model().embed_documents(texts)
    return [np.array(v, dtype=np.float32) for v in vectors]


@lru_cache
def get_psycopg_connection():
    """
    psycopg3 연결 (dict_row row_factory).
    pgvector 타입 어댑터를 등록해 vector 파라미터 바인딩이 가능하게 함.
    """
    s = get_settings()["postgres"]
    conn = psycopg.connect(
        host=s["host"],
        port=s["port"],
        user=s["user"],
        password=s["password"],
        dbname=s["dbname"],
        row_factory=dict_row,
    )
    register_vector(conn)
    return conn

