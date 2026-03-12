"""
시장성 평가용 PDF 적재 스크립트

사용법:
    python ingestion.py "../docs/공표_2024년 기준 로봇산업 실태조사-페이지-2.pdf"
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg
from langchain_community.embeddings import HuggingFaceEmbeddings
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings


# ============================================================
# 상수 정의
# ============================================================


EMBEDDING_MODEL = "jhgan/ko-sbert-nli"


@dataclass(frozen=True)
class PageText:
    """PDF 페이지 단위 텍스트."""

    page: int
    text: str


# ============================================================
# PDF 로드
# ============================================================


def load_pdf_pages(pdf_path: str) -> List[PageText]:
    """PDF에서 페이지 단위 텍스트를 추출한다."""

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise SystemExit(
            "pypdf가 설치되어 있지 않습니다. "
            "`.venv`에서 `pip install pypdf` 또는 `pip install -r requirements.txt`를 먼저 실행하세요."
        ) from exc

    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    pages: List[PageText] = []

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if not text:
            continue
        pages.append(PageText(page=page_index, text=text))

    return pages


def load_pdf_text(pdf_path: str) -> str:
    """PDF에서 텍스트를 추출해 하나의 문자열로 합친다."""

    parts: List[str] = []
    for item in load_pdf_pages(pdf_path):
        parts.append(f"\n\n--- page {item.page} ---\n\n{item.text}")
    return "".join(parts).strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """단순 문자 기준 청킹."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    source = (text or "").strip()
    if not source:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(source):
        end = min(len(source), start + chunk_size)
        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(source):
            break
        start = max(0, end - chunk_overlap)

    return chunks


def split_pages_into_chunks(
    pages: List[PageText],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Tuple[str, Dict[str, Any]]]:
    """페이지 번호를 유지한 채 chunk와 metadata를 만든다."""

    chunk_items: List[Tuple[str, Dict[str, Any]]] = []
    chunk_index = 0

    for item in pages:
        page_chunks = split_text_into_chunks(
            item.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for content in page_chunks:
            chunk_items.append(
                (
                    content,
                    {
                        "page": item.page,
                        "chunk_index": chunk_index,
                    },
                )
            )
            chunk_index += 1

    return chunk_items


# ============================================================
# 임베딩 / DB 연결
# ============================================================


@lru_cache
def get_embedding_model() -> HuggingFaceEmbeddings:
    """시장 자료 적재에 사용할 임베딩 모델."""

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """문서 chunk들을 임베딩한다."""

    vectors = get_embedding_model().embed_documents(texts)
    return [np.array(vector, dtype=np.float32) for vector in vectors]


@lru_cache
def get_psycopg_connection():
    """pgvector 타입 등록이 된 psycopg 연결을 반환한다."""

    settings = get_settings()["postgres"]
    conn = psycopg.connect(
        host=settings["host"],
        port=settings["port"],
        user=settings["user"],
        password=settings["password"],
        dbname=settings["dbname"],
    )
    register_vector(conn)
    return conn


# ============================================================
# 적재 함수
# ============================================================


def ingest_market_pdf(
    pdf_path: str,
    *,
    source: str = "market_report",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """시장 보고서 PDF를 market_eval vector store에 적재한다."""

    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = load_pdf_pages(str(path))
    chunk_items = split_pages_into_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunk_items:
        return 0

    settings = get_settings()
    schema_name = settings["vectorstore"]["schema_name"]
    table_name = settings["vectorstore"]["market_eval_table"]
    texts = [content for content, _ in chunk_items]
    vectors = embed_texts(texts)
    conn = get_psycopg_connection()

    delete_sql = sql.SQL("DELETE FROM {}.{} WHERE metadata->>'source_file' = %s").format(
        sql.Identifier(schema_name),
        sql.Identifier(table_name),
    )
    insert_sql = sql.SQL(
        "INSERT INTO {}.{} (content, metadata, embedding) VALUES (%s, %s, %s)"
    ).format(sql.Identifier(schema_name), sql.Identifier(table_name))

    rows: List[Any] = []
    for (content, metadata), vector in zip(chunk_items, vectors):
        row_metadata: Dict[str, Any] = {
            "source": source,
            "source_file": path.name,
        }
        row_metadata.update(metadata)
        rows.append((content, Jsonb(row_metadata), vector))

    with conn.cursor() as cursor:
        cursor.execute(delete_sql, (path.name,))
        cursor.executemany(insert_sql, rows)
    conn.commit()

    return len(rows)


# ============================================================
# CLI
# ============================================================


def main() -> int:
    """CLI 진입점."""

    if len(sys.argv) < 2:
        raise SystemExit(
            "사용법: python ingestion.py "
            "\"../docs/공표_2024년 기준 로봇산업 실태조사-페이지-2.pdf\""
        )

    try:
        inserted = ingest_market_pdf(sys.argv[1])
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"적재 완료: {inserted} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
