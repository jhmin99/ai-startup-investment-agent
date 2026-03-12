"""
PostgreSQL + pgvector 설정.
.env 또는 환경 변수에서 DB 연결 정보와 벡터 스토어 테이블 설정을 읽습니다.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@lru_cache
def get_settings():
    """환경 변수 기반 설정 (캐시)."""
    return {
        "postgres": {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "dbname": os.getenv("POSTGRES_DB", "postgres"),
        },
        "vectorstore": {
            "schema_name": os.getenv("VECTORSTORE_SCHEMA", "public"),
            "startup_search_table": os.getenv(
                "VECTORSTORE_TABLE_STARTUP_SEARCH", "rag_vector_store_startup_search"
            ),
            "market_eval_table": os.getenv(
                "VECTORSTORE_TABLE_MARKET_EVAL", "rag_vector_store_market_eval"
            ),
        },
    }


def get_postgres_url(driver: str = "psycopg") -> str:
    """
    PostgreSQL 연결 URL 반환.
    driver: "psycopg" (sync, psycopg3) 또는 "asyncpg" (async)
    """
    s = get_settings()["postgres"]
    # postgresql+psycopg:// 또는 postgresql+asyncpg://
    return (
        f"postgresql+{driver}://{s['user']}:{s['password']}"
        f"@{s['host']}:{s['port']}/{s['dbname']}"
    )
