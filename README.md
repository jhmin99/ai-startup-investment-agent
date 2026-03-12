# ai-startup-investment-agent

Multi-agent system using LangGraph and Agentic RAG to evaluate AI startup investment opportunities (Robotics 도메인).

---

## 1. 가상환경 설정

```bash
# 가상환경 생성
python3 -m venv .venv

# 활성화 (macOS/Linux)
source .venv/bin/activate

# 활성화 (Windows)
# .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 1.1 (macOS) PDF 출력(WeasyPrint) 시스템 의존성

`agents/final_report_agent`에서 HTML → PDF 변환은 `weasyprint`를 사용합니다.  
macOS에서는 파이썬 패키지 외에 **시스템 라이브러리(pango/cairo/glib 등)** 가 필요할 수 있습니다.

```bash
brew install pango cairo gdk-pixbuf libffi glib
```

---

## 2. RAG 환경 구성

### 2.1 PostgreSQL + pgvector

- **DB**: PostgreSQL with pgvector extension
- 로컬 예시: `docker run -it --rm -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres pgvector/pgvector`

### 2.2 환경 변수

`.env.example`을 복사해 `.env`를 만들고 값 채우기:

```bash
cp .env.example .env
```

- **RAG 임베딩**: 로컬 `jhgan/ko-sbert-nli` (한국어, 768차원) — API 키 불필요
- **PostgreSQL**: `POSTGRES_*` (호스트/포트/유저/비밀번호/DB명) 필수  
- `OPENAI_API_KEY`: LLM(채팅 등) 사용 시에만 필요

### 2.3 DB 스키마 초기화

임베딩 차원 768 (`jhgan/ko-sbert-nli`). 에이전트별 RAG 벡터 스토어 테이블 2개 + 문서 메타 1개:

| 테이블 | 용도 |
|--------|------|
| `rag_vector_store_startup_search` | 스타트업 탐색 에이전트 (벤처기업 데이터 등) |
| `rag_vector_store_market_eval` | 시장성 평가 에이전트 (로봇산업 실태조사, 시장 보고서 등) |
| `documents` | 문서 메타데이터 (`store_type`으로 어느 스토어인지 구분) |

```bash
psql -U postgres -h localhost -d postgres -f sql/init.sql
```

### 2.4 RAG 사용 에이전트와의 매핑

| 에이전트 | RAG 테이블 | 용도 |
|----------|------------|------|
| **스타트업 탐색** | `rag_vector_store_startup_search` | 벤처기업 데이터 기반 후보 탐색 |
| **시장성 평가** | `rag_vector_store_market_eval` | 시장·산업 보고서 기반 분석 |
| 질문 정제 / 기술 요약 / 투자 판단 / 웹 검색 / 최종 보고서 | - | RAG 미사용 |

**사용 예**

```python
from vectorstore import (
    get_vector_store,
    get_vector_store_async,
    AGENT_STARTUP_SEARCH,
    AGENT_MARKET_EVAL,
)

# 스타트업 탐색 에이전트용 (동기)
store = get_vector_store(AGENT_STARTUP_SEARCH)

# 시장성 평가 에이전트용 (비동기)
store = await get_vector_store_async(AGENT_MARKET_EVAL)
```

---

## 3. 에이전트 구성 (참고)

- **Supervisor**: 하위 에이전트 실행 순서·분기, 최종 보고서까지 흐름 제어
- **스타트업 탐색 (RAG)**: Robotics AI 스타트업 후보 수집·선정
- **질문 정제**: 모호한 질의 → 정제된 검색 질의
- **기술 요약**: 스타트업 핵심 기술·차별성·한계 요약
- **시장성 평가 (RAG)**: TAM/SAM/SOM, 성장성, 확장성 분석
- **투자 판단**: 항목별 점수·총점·투자 추천 여부
- **웹 검색**: 부족 정보 보완용 웹 검색
- **최종 보고서 생성**: 기술/시장/판단 결과 종합 보고서 작성
