"""
PostgreSQL + pgvector 벡터 스토어 연동.
에이전트별 테이블: 스타트업 탐색 / 시장성 평가.
임베딩: jhgan/ko-sbert-nli (768차원).
"""

from langchain_postgres import PGEngine, PGVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import get_postgres_url, get_settings

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"

# 에이전트별 테이블 키 (get_vector_store(agent=...) 에 사용)
AGENT_STARTUP_SEARCH = "startup_search"
AGENT_MARKET_EVAL = "market_eval"


def get_pg_engine(sync: bool = True):
    """PGEngine 인스턴스 반환."""
    driver = "psycopg" if sync else "asyncpg"
    return PGEngine.from_connection_string(url=get_postgres_url(driver))


def _table_for_agent(agent: str) -> str:
    vs = get_settings()["vectorstore"]
    if agent == AGENT_STARTUP_SEARCH:
        return vs["startup_search_table"]
    if agent == AGENT_MARKET_EVAL:
        return vs["market_eval_table"]
    raise ValueError(f"Unknown agent: {agent}. Use AGENT_STARTUP_SEARCH or AGENT_MARKET_EVAL.")


def get_vector_store(agent: str, embedding=None, sync: bool = True):
    """
    에이전트별 PGVectorStore 반환.
    agent: AGENT_STARTUP_SEARCH (스타트업 탐색) | AGENT_MARKET_EVAL (시장성 평가)
    """
    schema_name = get_settings()["vectorstore"]["schema_name"]
    table_name = _table_for_agent(agent)
    engine = get_pg_engine(sync=sync)
    if embedding is None:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if not sync:
        raise ValueError("비동기 사용 시 get_vector_store_async() 를 사용하세요.")
    return PGVectorStore.create_sync(
        engine=engine,
        table_name=table_name,
        schema_name=schema_name,
        embedding_service=embedding,
        id_column="id",
        content_column="content",
        embedding_column="embedding",
        metadata_json_column="metadata",
    )


async def get_vector_store_async(agent: str, embedding=None):
    """에이전트별 비동기 PGVectorStore (LangGraph 등)."""
    schema_name = get_settings()["vectorstore"]["schema_name"]
    table_name = _table_for_agent(agent)
    engine = get_pg_engine(sync=False)
    if embedding is None:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return await PGVectorStore.create(
        engine=engine,
        table_name=table_name,
        schema_name=schema_name,
        embedding_service=embedding,
        id_column="id",
        content_column="content",
        embedding_column="embedding",
        metadata_json_column="metadata",
    )

