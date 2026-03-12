-- PostgreSQL + pgvector 초기 스키마
-- RAG 에이전트별 벡터 스토어 테이블 2개 + 문서 메타데이터 1개

-- 필수 확장
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 스타트업 탐색 에이전트용 (벤처기업 데이터 등 후보 탐색)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.rag_vector_store_startup_search (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768)   -- jhgan/ko-sbert-nli
);

CREATE INDEX IF NOT EXISTS rag_vector_store_startup_search_embedding_idx
    ON public.rag_vector_store_startup_search
    USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- 시장성 평가 에이전트용 (로봇산업 실태조사, 시장 보고서 등)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.rag_vector_store_market_eval (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding vector(768)   -- jhgan/ko-sbert-nli
);

CREATE INDEX IF NOT EXISTS rag_vector_store_market_eval_embedding_idx
    ON public.rag_vector_store_market_eval
    USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- 문서 메타데이터 (어느 스토어에 넣었는지 store_type으로 구분)
-- ============================================================
CREATE TABLE IF NOT EXISTS public.documents (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    store_type VARCHAR(64) NOT NULL,  -- 'startup_search' | 'market_eval'
    source VARCHAR(512),
    filename VARCHAR(255),
    content_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS documents_store_type_idx ON public.documents (store_type);
CREATE INDEX IF NOT EXISTS documents_created_at_idx ON public.documents (created_at);
CREATE INDEX IF NOT EXISTS documents_content_hash_idx ON public.documents (content_hash);
