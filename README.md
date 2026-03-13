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
 
- **Tools**
  - Tavily API, OpenAI, pypdf

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
| Web Search  | Tavily API + OpenAI                                   |
| Report      | Custom HTML/CSS, WeasyPrint (`html_to_pdf`)                      |

---

## 3. System Architecture
그림

## 4. Agents

- **Query Refinement Agent**  
  사용자 질의를 투자 평가에 적합한 형태로 정제 (모호한 표현 및 오타 제거, 핵심 키워드 추출)
  정제된 질의는 Supervisor에게 전달

- **Startup Search Agent (RAG)**  
  - 정제된 질의를 기반으로 PDF 기반 데이터를 적재한 pgvector에서 유사도 검색  
  - 회사별로 chunk를 병합하고, 다음 항목을 **스키마 기반으로 구조화**:
    - 회사 개요 (대표자, 설립연도, 지역, 업종, 주생산품, 투자 현황, 웹사이트)
    - 기술 요약 (기술 성숙도, 특허/IP)
    - 기술적 강점/한계
    - 경쟁사 대비 차별성 + 핵심 경쟁 우위
    - 실적 (고객/레퍼런스, 매출/성장, 도입 실적)
    - 참고 자료
  - 결과는 Supervisor에게 전달

- **Technical Summary Agent**  
  각 스타트업의 기술 정보를 LLM에 전달  
  투자 관점에서 
- 중심 기술력, 강점, 한계, 차별성, 특허 수, R&D 규모 등을 집중 요약
  -  결과는 Supervisor에게 전달

- **Market Evaluation Agent (RAG)**  
  - 정제된 질의 + 회사 맥락을 바탕으로  시장보고서 PDF를 적재한 pgvector에서 evidence 조회   
  - LLM으로 시장성 평가 JSON 생성
    - `market_summary`, `market_size`
    - `growth_drivers`
    - `target_industries` (주요 수요 산업)
    - `competition_analysis`, `customer_adoption`
    - `key_risks`, `evidence`
- 결과는 Supervisor에게 전달

- **Web Search Agent**  
  - 각 스타트업 이름으로 Tavily + OpenAI를 활용한 웹 검색 수행  
  - 카테고리별 팩트 추출:
    - `market` (시장 규모/성장)
    - `technology` (기술/제품 기사)
    - `competition` (경쟁사/파트너십)
    - `performance` (투자 유치, 매출, 도입 사례 등)
  - 결과는 Supervisor에게 전달

- **Investment Decision Agent**  
  - 기술 요약 + 시장성 평가 + 웹 검색 결과를 종합
    - 4개 영역(시장, 기술, 경쟁력, 실적)에 대해 1~5점 스코어 산출
    - 가중합 총점(0~100)과 최종 verdict:
      - **적극 투자 / 투자 가능 / 보류 / 투자 위험**
- 결과는 Supervisor에게 전달

- **draft Report Agent**  
  - 상기 모든 결과를 하나의 **초안 리포트**로 뼈대 구축
    - 헤더: 회사명, 종합 점수, 최종 의견
    - Summary Strip: 핵심 사업, 핵심 장점, 주요 리스크 태그
    - 기업 소개, 기술 분석, 시장 분석, 위험 요소, 경쟁 분석, 투자 평가 카드
    - 한계점(제한 사항), 참고 자료(RAG/Web) 리스트  
- 결과는 Supervisor에게 전달

---

## 5. Directory Structure

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
├── sql/init.sql                    # PostgreSQL + pgvector 스키마 초기화 스크립트
├── vectorstore.py                  # PGVector Store 팩토리 (startup_search, market_eval)
├── config.py                       # 환경변수 기반 설정 로더
├── requirements.txt
├── README.md
└── report.html                     # 최근 생성된 리포트 (출력 예시)
```

---

## 6. Contributors

- **김나령**: 질문 정제 에이전트, 시장성 평가 에이전트 구현, 그래프 아키텍처 설계
- **민지홍**: 프로젝트 설계, Supervisor Graph, 스타트업 탐색/보고서 통합  
- **백강민**: 투자 판단 에이전트, 최종 스코어링/판정 로직  
- **임진영**: 기술 요약 에이전트, 웹 검색 에이전트, 리포트 템플릿 기획  

