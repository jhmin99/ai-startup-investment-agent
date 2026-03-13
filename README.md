# AI Startup Investment Evaluation Agent

본 프로젝트는 **로보틱스/AI 스타트업**에 대한 투자 가능성을 자동으로 평가하는 **멀티‑에이전트 투자 평가 시스템**입니다.  
PDF 기업 정보 + 공공 시장 보고서 + 웹 검색을 조합하여, 최종적으로 **최종 투자 판단 리포트**를 생성합니다.

---

## 1. Overview

- **Objective**  
  로봇/AI 스타트업의 **기술력, 시장성, 경쟁력, 실적**을 정량·정성적으로 평가하고,  
  투자 관점에서 **적극 투자 / 투자 가능 / 보류 / 투자 위험**을 자동 판정.

- **Method**  
  - LangGraph 기반 **Supervisor Agent**가 여러 하위 에이전트를 오케스트레이션
  - **RAG + Web Search + Market Evaluation + LLM 스코어링** 조합
  - 모든 단계의 근거(원문 텍스트/URL)를 최대한 보존하여 **근거 기반 평가** 지향

- **Input**
  - 로봇 스타트업 기업 정보 PDF (`docs/로봇_스타트업_기업정보.pdf`)
  - 로봇산업 실태조사 등 시장 보고서 PDF (`docs/공표_2024년 기준 로봇산업 실태조사-페이지-2.pdf`)
  - 사용자 질의 (예: `"물류 자동화 로봇 스타트업 투자 기회 평가해줘"`)

- **Output**
  - 스타트업별 상세 분석이 포함된 **HTML 투자 보고서** PDF 변환

---

## 2. Tech Stack

| Category     | Details                                                                 |
|-------------|-------------------------------------------------------------------------|
| Language    | Python 3.9                                                              |
| Framework   | LangGraph, LangChain, Pydantic                                         |
| LLM         | OpenAI GPT‑4o / GPT‑4o‑mini via `langchain-openai`, custom JSON client |
| Retrieval   | PostgreSQL + pgvector (`rag_vector_store_startup_search`, `market_eval`) |
| Embedding   | `jhgan/ko-sbert-nli` (Korean SBERT, 768‑dim)                            |
| PDF Parsing | `pypdf`                                                                 |
| Web Search  | Tavily API (`tavily-python`) + OpenAI                                   |
| Report      | Custom HTML/CSS, (옵션) WeasyPrint (`html_to_pdf`)                      |

---

## 3. System Architecture

### 3.1 High‑Level Flow

- **Query Refinement Agent**  
  사용자 질의를 투자 평가에 적합한 형태로 정제 (모호한 표현 제거, 핵심 키워드 추출)

- **Startup Search Agent (RAG)**  
  - `docs/로봇_스타트업_기업정보.pdf` 를 PDF → chunk → pgvector에 적재  
  - 정제된 질의를 기반으로 유사도 검색  
  - 회사별로 chunk를 병합하고, 다음 항목을 **스키마 기반으로 구조화**:
    - 회사 개요 (대표자, 설립연도, 지역, 업종, 주생산품, 투자 현황, 웹사이트)
    - 기술 요약 (기술 성숙도, 특허/IP)
    - 기술적 강점/한계
    - 경쟁사 대비 차별성 + 핵심 경쟁 우위
    - 실적 (고객/레퍼런스, 매출/성장, 도입 실적)
    - 참고 자료(문서 내 `참고:` URL)
  - 결과는 `StartupSearchOutput.startup_profiles` 리스트로 Supervisor에 전달

- **Technical Summary Agent**  
  각 스타트업의 구조화된 기술 정보 + raw 텍스트를 LLM에 전달  
  투자 관점에서:
  - `core_technology`, `tech_summary`
  - `strengths`, `limitations`, `differentiation`
  - 특허 수, R&D 규모 등  
  LLM JSON 출력 → Supervisor에서 공통 스키마(`technology_summary`)로 정규화

- **Web Search Agent**  
  - 각 스타트업 이름으로 Tavily + OpenAI를 활용한 웹 검색 수행  
  - 카테고리별 팩트 추출:
    - `market` (시장 규모/성장)
    - `technology` (기술/제품 기사)
    - `competition` (경쟁사/파트너십)
    - `performance` (투자 유치, 매출, 도입 사례 등)
  - 결과는 `web_search_results`로 Supervisor에 전달

- **Market Evaluation Agent (RAG)**  
  - `rag_vector_store_market_eval` (시장 보고서 RAG)
  - 정제된 질의 + 회사 맥락(업종/주생산품/기술 요약)을 바탕으로 시장 보고서에서 evidence 조회  
  - LLM으로 시장성 평가 JSON 생성:
    - `market_summary`, `market_size`
    - `growth_drivers`
    - `target_industries` (주요 수요 산업)
    - `competition_analysis`, `customer_adoption`
    - `key_risks`, `evidence`

- **Investment Decision Agent**  
  - 기술 요약 + 시장성 평가 + 웹 검색 결과를 종합해서:
    - 4개 영역(시장, 기술, 경쟁력, 실적)에 대해 1~5점 스코어 산출 (LLM 또는 휴리스틱)
    - 가중합 총점(0~100)과 최종 verdict:
      - **적극 투자 / 투자 가능 / 보류 / 투자 위험**
    - 부족한 정보 목록(`missing_information`)과 Web Search 필요 여부 판단

- **Final Report Agent**  
  - 상기 모든 결과를 하나의 **고급 HTML 리포트**로 렌더링:
    - 헤더: 회사명, 종합 점수, 최종 의견
    - Summary Strip: 핵심 사업, 핵심 장점, 주요 리스크 태그
    - 기업 소개, 기술 분석, 시장 분석, 위험 요소, 경쟁 분석, 투자 평가 카드
    - 한계점(제한 사항), 참고 자료(RAG/Web) 리스트  
  - 여러 회사가 선택된 경우, 투자 점수 기준 **내림차순**으로 정렬하여  
    한 페이지에서 **회사 1~N 보고서**를 연속 섹션으로 출력

---

## 4. Directory Structure

```text
.
├── agents
│   ├── __init__.py
│   ├── startup_search/             # 스타트업 탐색(RAG) 에이전트
│   ├── query_refinement/           # 질문 정제 에이전트
│   ├── technical-summary/          # 기술 요약 에이전트
│   ├── web-search/                 # 웹 검색 에이전트
│   ├── market_evaluation_agent/    # 시장성 평가 에이전트 (market_eval RAG)
│   ├── investment_decision_agent/  # 투자 판단 에이전트 (LLM/휴리스틱 스코어링)
│   ├── final_report_agent/         # 최종 HTML/PDF 리포트 생성 에이전트
│   └── supervisor/                 # LangGraph Supervisor (에이전트 오케스트레이션)
├── docs/                           # PDF 데이터 (기업 정보, 시장 보고서 등)
├── market-evaluation-agent/        # (원본) 시장성 평가 에이전트 단독 실행 버전
├── sql/init.sql                    # PostgreSQL + pgvector 스키마 초기화 스크립트
├── vectorstore.py                  # PGVector Store 팩토리 (startup_search, market_eval)
├── config.py                       # 환경변수 기반 설정 로더
├── requirements.txt
├── README.md
└── report.html                     # 최근 생성된 리포트 (출력 예시)
```

---

## 5. Setup

### 5.1 가상환경 설정

```bash
python3 -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 5.2 (macOS) PDF 출력(WeasyPrint) 시스템 의존성

`agents/final_report_agent`에서 HTML → PDF 변환은 `weasyprint`를 사용합니다.  
macOS에서는 파이썬 패키지 외에 **시스템 라이브러리(pango/cairo/glib 등)** 가 필요할 수 있습니다.

```bash
brew install pango cairo gdk-pixbuf libffi glib
```

### 5.3 환경 변수

`.env.example`을 복사해 `.env`를 만들고 값 채우기:

```bash
cp .env.example .env
```

주요 항목:

- **PostgreSQL**: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- **RAG 임베딩**: 로컬 `jhgan/ko-sbert-nli` (한국어, 768차원) — API 키 불필요
- **LLM**: `OPENAI_API_KEY`, `OPENAI_MODEL` (예: `gpt-4o-mini`)
- **Web Search**: `TAVILY_API_KEY`

### 5.4 DB 스키마 초기화

임베딩 차원 768 (`jhgan/ko-sbert-nli`). 에이전트별 RAG 벡터 스토어 테이블 2개 + 문서 메타 1개:

| 테이블 | 용도 |
|--------|------|
| `rag_vector_store_startup_search` | 스타트업 탐색 에이전트 (벤처기업 데이터 등) |
| `rag_vector_store_market_eval`   | 시장성 평가 에이전트 (로봇산업 실태조사, 시장 보고서 등) |
| `documents`                      | 문서 메타데이터 (`store_type`으로 어느 스토어인지 구분) |

```bash
cd /ai-startup-investment-agent

.venv/bin/python -c "from agents.startup_search.ingestion import ensure_db_initialized; \
ensure_db_initialized('sql/init.sql')"
```

> `init.sql`은 `rag_vector_store_startup_search`, `rag_vector_store_market_eval`을  
> `DROP TABLE IF EXISTS` 후 새로 생성합니다. 기존 인덱싱을 초기화하고 싶을 때만 실행하세요.

---

## 6. Data Ingestion

### 6.1 스타트업 기업 정보 PDF → startup_search 벡터 스토어

```bash
.venv/bin/python -c "from agents.startup_search.ingestion import embed_and_store; \
print(embed_and_store('docs/로봇_스타트업_기업정보.pdf'))"
```

### 6.2 시장 보고서 PDF → market_eval 벡터 스토어

```bash
.venv/bin/python market-evaluation-agent/ingestion.py \
  'docs/공표_2024년 기준 로봇산업 실태조사-페이지-2.pdf'
```

---

## 7. Running the Supervisor (End‑to‑End)

```bash
cd /ai-startup-investment-agent

.venv/bin/python -m agents.supervisor.run \
  "물류 자동화 로봇 스타트업 투자 기회 평가해줘" \
  --k 10 \
  --max-startups 5 \
  --output-html report.html
```

옵션:

- `--k`: startup_search 유사도 검색 상위 k (기본 10)
- `--max-startups`: downstream 에이전트에서 처리할 스타트업 수 (기본 3)
- `--no-web-search`: WebSearchAgent 비활성화
- `--no-tech-summary`: Technical Summary 비활성화

생성된 `report.html`을 브라우저에서 열면,  
**투자 점수 내림차순**으로 정렬된 여러 스타트업의 카드형 분석 리포트를 볼 수 있습니다.

---

## 8. Agents Summary

- **StartupSearchAgent (`agents/startup_search/`)**
  - PDF → pgvector embedding
  - 회사별 `StartupProfile` 구조화 + `StartupSearchOutput` 반환

- **QueryRefinementAgent (`agents/query_refinement/`)**
  - 사용자 질의를 정제하고 RAG/시장 평가에 적합한 형태로 변환

- **TechSummaryAgent (`agents/technical-summary/`)**
  - 기술 요약/강점/한계/차별성을 LLM으로 정리 (JSON 출력)

- **WebSearchAgent (`agents/web-search/`)**
  - Tavily + OpenAI로 최신 기사/투자/실적/경쟁사 정보 수집

- **MarketEvaluationAgent (`agents/market_evaluation_agent/`)**
  - `rag_vector_store_market_eval`에서 시장/산업 보고서 RAG 검색
  - 시장 규모, 성장 동인, 주요 수요 산업(`target_industries`), 리스크 등 평가

- **InvestmentDecisionAgent (`agents/investment_decision_agent/`)**

  **기본 조건 평가**
  - 시장 규모, 기술 차별성, 팀/경쟁 역량, 실적/재무 근거 충족 여부 판단

  **점수 평가**
  - 기술력 / 시장성 / 경쟁력 / 실적 4개 축 점수 산출
  - 가중합으로 **0~100점 총점** 및 최종 투자 verdict 산출
  - 부족한 정보(`missing_information`) 및 Web Search 필요 여부 판단

- **FinalReportAgent (`agents/final_report_agent/`)**
  - 모든 결과를 하나의 고급 HTML 투자 보고서로 렌더링
  - (옵션) WeasyPrint를 통해 PDF 출력

- **Supervisor (`agents/supervisor/`)**
  - LangGraph `StateGraph`로 전체 흐름 정의:
    - `query_refinement` → `startup_search` → (score check) → `technical_summary` → `web_search` → `market_evaluation` → `finalize`

---

## 9. Contributors

- **김나령**: 질문 정제 에이전트, 시장성 평가 에이전트 구현, 그래프 아키텍처 설계
- **민지홍**: 프로젝트 설계, Supervisor Graph, 스타트업 탐색/보고서 통합  
- **백강민**: 투자 판단 에이전트, 최종 스코어링/판정 로직  
- **임진영**: 기술 요약 에이전트, 웹 검색 에이전트, 리포트 템플릿 기획  

