from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader
from psycopg import sql
from psycopg.types.json import Jsonb

from config import get_settings

from .utils import embed_texts, extract_company_name_from_text, get_psycopg_connection
from .parser import extract_references


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


def load_pdf_text(pdf_path: str) -> str:
    """
    PDF에서 텍스트를 추출해 단일 문자열로 반환.

    - 한국어 PDF 특성상 extract_text()가 None이거나 예외가 날 수 있어 방어적으로 처리
    - 페이지 경계는 후속 처리를 위해 구분자를 삽입
    """
    pages = _load_pdf_pages(pdf_path)
    parts: List[str] = []
    for p in pages:
        parts.append(f"\n\n--- page {p.page} ---\n\n")
        parts.append(p.text)
    return "".join(parts).strip()


def _load_pdf_pages(pdf_path: str) -> List[PageText]:
    """페이지 단위 텍스트 추출 (내부용)."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(path))
    out: List[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # 특정 페이지에서 깨지더라도 전체 ingest가 중단되지 않게 함
            text = ""
        text = text.strip()
        if text:
            out.append(PageText(page=i, text=text))
    return out


def split_text_into_chunks(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[str]:
    """
    단순 문자 기준 청킹 (overlap 포함).
    나중에 기업 단위 청킹으로 교체 가능하도록 별도 함수로 유지.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    t = (text or "").strip()
    if not t:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(t):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _split_pages_into_chunks(
    pages: List[PageText], chunk_size: int, chunk_overlap: int
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    페이지 단위 텍스트를 chunk로 만들고,
    각 chunk에 들어갈 metadata를 함께 구성.
    """
    chunk_items: List[Tuple[str, Dict[str, Any]]] = []
    chunk_index = 0

    for p in pages:
        if not p.text.strip():
            continue

        # 페이지에 회사명 라인이 들어있는 경우가 많아서 metadata에 넣어둠
        company_name = extract_company_name_from_text(p.text)
        page_chunks = split_text_into_chunks(p.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in page_chunks:
            md: Dict[str, Any] = {
                "page": p.page,
                "chunk_index": chunk_index,
            }
            if company_name:
                md["company_name"] = company_name
            chunk_items.append((c, md))
            chunk_index += 1

    return chunk_items


def embed_and_store(
    pdf_path: str,
    source_file: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """
    PDF 텍스트 추출 → 청킹 → 임베딩(jhgan/ko-sbert-nli) → PostgreSQL(pgvector) 저장.

    저장 테이블: config.py의 vectorstore.startup_search_table
    저장 컬럼: content, metadata(jsonb), embedding(vector)

    metadata에 포함:
    - source_file
    - chunk_index
    - page
    - (가능하면) company_name
    """
    settings = get_settings()
    schema_name = settings["vectorstore"]["schema_name"]
    table_name = settings["vectorstore"]["startup_search_table"]

    pages = _load_pdf_pages(pdf_path)
    if not pages:
        return 0

    if source_file is None:
        source_file = Path(pdf_path).name

    # PDF 전체 텍스트에서 회사 공통 참고 URL(참고: 블록)을 한 번만 파싱
    full_text = "\n\n".join(p.text for p in pages)
    pdf_references = extract_references(full_text)

    chunk_items = _split_pages_into_chunks(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunk_items:
        return 0

    texts = [t for t, _ in chunk_items]
    vectors = embed_texts(texts)  # List[np.ndarray float32]

    conn = get_psycopg_connection()
    insert_sql = sql.SQL(
        "INSERT INTO {}.{} (content, metadata, embedding) VALUES (%s, %s, %s)"
    ).format(sql.Identifier(schema_name), sql.Identifier(table_name))

    rows = []
    for (content, md), vec in zip(chunk_items, vectors):
        md = dict(md)
        md["source_file"] = source_file
        # PDF 단위로 추출한 참고 URL을 metadata에 추가해두면,
        # 나중에 회사별 profile을 만들 때 chunk에 상관없이 참고 자료를 복원할 수 있다.
        if pdf_references:
            md["references"] = pdf_references
        rows.append((content, Jsonb(md), vec))

    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)
    conn.commit()

    return len(rows)


def ensure_db_initialized(init_sql_path: str = "sql/init.sql") -> None:
    """
    서버 부팅 시 init.sql을 자동 적용하기 위한 헬퍼.

    주의:
    - CREATE EXTENSION / CREATE TABLE을 포함하므로 DB 권한이 필요합니다.
    - init.sql은 idempotent( IF NOT EXISTS )라 반복 실행해도 안전합니다.
    """
    path = Path(init_sql_path)
    if not path.exists():
        raise FileNotFoundError(f"init.sql not found: {init_sql_path}")

    sql_text = path.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql_text)

    conn = get_psycopg_connection()
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()


def _split_sql_statements(sql_text: str) -> List[str]:
    """
    매우 단순한 SQL statement splitter.
    init.sql은 단순 DDL이므로 이 수준으로 충분합니다.
    """
    out: List[str] = []
    buff: List[str] = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buff.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buff).strip()
            if stmt:
                out.append(stmt)
            buff = []
    # 마지막에 ; 없는 경우도 처리
    tail = "\n".join(buff).strip()
    if tail:
        out.append(tail)
    return out


def is_startup_store_empty() -> bool:
    """startup_search 벡터 테이블에 데이터가 없는지 확인."""
    settings = get_settings()
    schema_name = settings["vectorstore"]["schema_name"]
    table_name = settings["vectorstore"]["startup_search_table"]

    conn = get_psycopg_connection()
    stmt = sql.SQL("SELECT 1 FROM {}.{} LIMIT 1").format(
        sql.Identifier(schema_name), sql.Identifier(table_name)
    )
    with conn.cursor() as cur:
        row = cur.execute(stmt).fetchone()
    return row is None
