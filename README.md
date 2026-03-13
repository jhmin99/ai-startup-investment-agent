# AI Startup Investment Evaluation Agent

본 프로젝트는 **로보틱스 스타트업**에 대한 투자 가능성을 자동으로 평가하는 **멀티‑에이전트 투자 평가 시스템**입니다.  
기업 정보 PDF + 공공 시장 보고서 + 웹 검색을 종합하여 최종적으로 **최종 투자 판단 리포트**를 생성합니다.

> 다양한 리소스에서 수집한 근거를 기반으로 투자 관점의 판단을 구조화하여 제공하는 **의사결정 보조 시스템**을 목표로 구현했습니다.

---

## 1. Overview

- **Objective**  
  로보틱스 도메인 스타트업의 **기술력, 시장성, 경쟁력, 실적**을 정량·정성적으로 평가하고
  투자 관점에서 **적극 투자 / 투자 가능 / 보류 / 투자 위험**을 판정하여
  투자 판단을 위한 보고서를 생성하는 것을 목표로 합니다.

- **Method**  
  - LangGraph 기반 **Supervisor Agent**가 여러 하위 에이전트를 오케스트레이션
  - **RAG + Web Search + Market Evaluation + LLM 스코어링** 조합
  - 기업 정보 PDF와 시장 보고서를 각각 검색하여 **정적 정보와 산업 맥락**을 확보
  - 웹 검색을 통해 **최신 기사, 투자 유치, 파트너십, 도입 사례 등 최신성 정보**를 보완
  - 모든 단계의 근거(원문 텍스트/URL)를 최대한 보존하여 **근거 기반 평가** 지향

- **Input**
  - 로봇 스타트업 기업 정보 PDF (`docs/로봇_스타트업_기업정보.pdf`)
  - 로봇산업 실태조사 등 시장 보고서 PDF (`docs/공표_2024년 기준 로봇산업 실태조사-페이지-2.pdf`)
  - 사용자 질의 (예: `"물류 자동화 로봇 스타트업 투자 기회 평가해줘"`)

- **Output**
  - 스타트업별 상세 분석이 포함된 **투자 보고서** PDF 변환
  - 리포트에는 기업 개요, 기술 분석, 시장 분석, 경쟁력 평가, 실적 평가, 주요 리스크, 최종 투자 의견, 참고 근거가 포함됩니다.
 
---

## 2. Tech Stack

| Category     | Details                                                                 | Purpose |
|-------------|-------------------------------------------------------------------------|---------|
| Language    | Python 3.9                                                              | 전체 시스템 구현 |
| Framework   | LangGraph, LangChain, Pydantic                                          | 에이전트 오케스트레이션 및 스키마 기반 출력 |
| LLM         | OpenAI GPT-4o / GPT-4o-mini via `langchain-openai`, custom JSON client  | 요약, 구조화, 평가, 스코어링 |
| Retrieval   | PostgreSQL + pgvector (`rag_vector_store_startup_search`, `market_eval`) | 기업 정보 / 시장 보고서 검색 |
| Embedding   | `jhgan/ko-sbert-nli` (Korean SBERT, 768-dim)                            | 한국어 문서 임베딩 |
| PDF Parsing | `pypdf`                                                                 | PDF 텍스트 추출 |
| Web Search  | Tavily API + OpenAI                                                     | 최신 정보 및 외부 검증 근거 확보 |
| Report      | Custom HTML/CSS, WeasyPrint (`html_to_pdf`)                             | 최종 투자 리포트 생성 및 PDF 변환 |

---

## 3. System Architecture
<img width="1357" height="326" alt="AI_START_RE drawio" src="https://github.com/user-attachments/assets/a052e5ab-d7e4-4c5d-921b-0cb30e2fb5e6" />

본 시스템은 **Supervisor Agent 중심의 멀티-에이전트 오케스트레이션 구조**로 설계되었습니다.  
사용자 질의는 먼저 **Supervisor**에게 전달되며 Supervisor는 질의의 목적과 현재 수집된 정보 상태에 따라
적절한 하위 에이전트를 선택하고 호출한 뒤 최종적으로 응답과 투자 판단 리포트를 생성합니다.

### Architecture Summary
- **Query → Supervisor → Response**의 상위 흐름을 가집니다.
- Supervisor는 각 하위 에이전트의 실행을 조정하며 필요한 정보를 순차적 또는 조합적으로 수집합니다.
- 각 에이전트는 특정 역할에 특화되어 있으며 단일 에이전트가 모든 작업을 수행하지 않고 역할을 분리하여 처리합니다.
- 최종적으로 Supervisor가 각 에이전트의 결과를 통합해 투자 판단 및 보고서 초안을 생성합니다.

## 4. Agents

### Agent Orchestration
Supervisor는 다음 에이전트들을 오케스트레이션합니다.

| Agent | Main Role | Method | Data Source / Publisher |
|---|---|---|---|
| Query Refinement Agent | 사용자 질의를 투자 평가에 적합한 형태로 정제 | 질의 정제, 키워드 추출, 검색 질의 재구성 | 사용자 입력 질의 |
| Startup Search Agent | 로보틱스/AI 스타트업 후보를 탐색하고 기업 기본 정보를 구조화 | **RAG Retrieval** (pgvector 유사도 검색, 기업 단위 chunk 병합) | [중소벤처기업부 벤처확인종합관리시스템], [공공데이터포털 벤처기업 데이터] |
| Technical Summary Agent | 스타트업의 핵심 기술, 강점, 한계, 차별성을 투자 관점에서 요약 | LLM 기반 기술 분석 및 요약 | Startup Search Agent가 확보한 기업 정보 및 관련 문서 |
| Market Evaluation Agent | 스타트업이 속한 시장의 성장성, 수요, 확장성을 평가 | **RAG Retrieval + LLM 구조화** | [한국로봇산업진흥원 로봇산업 실태조사], [공공데이터포털 산업 자료] |
| Web Search Agent | 최신 외부 정보와 추가 근거를 보완 | Tavily API + OpenAI 기반 웹 검색 | 기업 공식 홈페이지, 보도자료, 투자 기사, 산업 뉴스, 파트너십/고객사 관련 외부 웹 문서 |
| Investment Decision Agent | 기술력, 시장성, 경쟁력, 실적을 종합해 최종 투자 판단 수행 | LLM 기반 스코어링 + 휴리스틱 가중합 | 기술 요약 결과, 시장성 평가 결과, 스타트업 탐색 결과, 웹 검색 보완 정보 |
| Report Draft Agent | 전체 결과를 투자 보고서 형식으로 재구성 | HTML/CSS 템플릿 기반 리포트 생성 | 스타트업 기본 정보, 기술 요약 결과, 시장성 평가 결과, 투자 판단 결과, 참고 자료 목록 |

## References

- [중소벤처기업부 벤처확인종합관리시스템](https://www.smes.go.kr/venturein/)
- [공공데이터포털 벤처기업 확인 정보](https://www.data.go.kr/)
- [한국로봇산업진흥원](https://www.kiria.org/)
- [공공데이터포털](https://www.data.go.kr/)

---
## 5. Evaluation Criteria

본 프로젝트의 평가 기준은 초기 스타트업 투자에서 널리 활용되는 엔젤 투자 Scorecard Method를 참고해 설계했습니다.
해당 방법론은 시장 기회, 제품/기술, 경쟁 우위, 실적 등 다양한 요소를 종합적으로 평가한다는 점에서 본 시스템의 투자 판단 목적과 부합하기 때문입니다. 
이에 따라 본 프로젝트에서는 로보틱스 스타트업에 적합하도록 평가 항목을 재구성하고 각 항목의 중요도를 반영해 가중치를 설정했습니다.

| 평가 항목 | 설명 | 비중 |
|---|---|---:|
| 시장성 | 시장 규모와 향후 성장 가능성을 평가, 각 스타트업이 속한 산업의 확장성과 투자 매력을 분석 | 35% |
| 기술력 | 보유 기술의 수준, 차별성, 특허 및 기술적 완성도를 바탕으로 기업의 기술 경쟁력을 평가 | 25% |
| 경쟁력 | 경쟁사 대비 차별화된 강점과 시장 내 우위를 확보할 수 있는 요소를 중심으로 기업의 경쟁력을 평가 | 20% |
| 실적 | 고객 확보 현황, 매출 성과, 투자 유치 이력 등을 바탕으로 기업의 사업성과와 성장 기반을 평가 | 20% |
---

## 6. Directory Structure

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

## 7. Sample Report

본 시스템이 생성한 최종 산출물 예시는 아래 PDF에서 확인할 수 있습니다.

- [최종 투자 보고서 예시 (PDF)](./finald_report_20260313.pdf)


## 8. Contributors

| 이름 | 담당 역할 |
|---|---|
| 김나령 | 질문 정제 에이전트 구현, 시장성 평가 에이전트 구현, 아키텍처 설계 |
| 민지홍 | 프로젝트 설계, Supervisor 구현, 스타트업 탐색 및 보고서 통합 구현|
| 백강민 | 투자 판단 에이전트 구현, 최종 스코어링 및 판단  설계 |
| 임진영 | 기술 요약 에이전트 구현, 웹 검색 에이전트 구현, 리포트 템플릿 기획 |

