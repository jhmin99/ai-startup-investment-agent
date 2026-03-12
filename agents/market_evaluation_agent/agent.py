"""
시장성 평가 에이전트 - agents/market_evaluation_agent 버전.

상위 그래프에서:
    from agents.market_evaluation_agent import MarketEvaluationAgent
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .prompts import MARKET_EVALUATION_PROMPT
from .state import MarketEvaluationOutput


DEFAULT_MODEL = "gpt-4o"
DEFAULT_TOP_K = 4
FETCH_MULTIPLIER = 4
DEFAULT_TIMEOUT_SECONDS = 30.0


REPORT_CATEGORY_RULES = [
    (["스마트팩토리", "제조", "생산라인", "공정", "전장", "전자"], ["제조업용 로봇", "로봇시스템"]),
    (["건설", "양중", "도장", "프린팅"], ["전문서비스용 로봇", "서비스용 로봇"]),
    (["물류", "배송", "병원", "호텔", "리테일", "자율주행"], ["전문서비스용 로봇", "서비스용 로봇", "로봇서비스"]),
    (["개인", "가정", "돌봄", "청소"], ["개인서비스용 로봇", "서비스용 로봇"]),
]


MARKET_EVALUATION_SYSTEM_PROMPT = """너는 Robotics 스타트업 투자 검토를 위한 시장성 평가 분석가다.
반드시 검색 근거만 바탕으로 평가하고, JSON object만 반환한다.
근거가 약하면 과장하지 말고 부족하다고 명시한다.
"""


class MarketEvaluationLLMClient(Protocol):
    """시장성 평가에 필요한 최소 LLM 인터페이스."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """모델이 생성한 JSON 객체를 반환한다."""


@dataclass
class OpenAIChatClient:
    """OpenAI 호환 Chat Completions API 클라이언트."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(
        cls,
        *,
        openai_api_key: Optional[str] = None,
        default_model: str = DEFAULT_MODEL,
    ) -> "OpenAIChatClient":
        import os

        api_key = (openai_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError(
                "시장성 평가는 LLM 기반으로만 동작합니다. "
                "OPENAI_API_KEY를 설정한 뒤 다시 실행하세요."
            )

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", default_model),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        import json
        from urllib import error, request

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenAI Chat API 호출 실패: {exc} {detail}") from exc

        obj = json.loads(data)
        try:
            content = obj["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"예상치 못한 OpenAI 응답 형식: {obj}") from exc

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"모델 응답을 JSON으로 파싱하지 못했습니다: {content}") from exc


class MarketEvaluationAgent:
    """
    시장 자료 기반으로 스타트업의 시장성을 평가하는 에이전트.

    - 검색은 `vectorstore.py`의 market_eval 스토어만 사용한다.
    - 결과는 투자 판단 단계로 넘길 구조화된 시장성 요약만 반환한다.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        vector_store: Optional[Any] = None,
        client: Optional[MarketEvaluationLLMClient] = None,
    ) -> None:
        self.model = model
        self.vector_store = vector_store
        self.client = client or OpenAIChatClient.from_env(
            openai_api_key=openai_api_key,
            default_model=model,
        )

    def __call__(self, state: Dict[str, Any]) -> MarketEvaluationOutput:
        """상위 그래프의 노드로 호출될 때 실행."""

        startup_name = str(state.get("startup_name") or "").strip()
        market_query = str(state.get("market_query") or "").strip()
        company_context = str(state.get("company_context") or "").strip()
        top_k = int(state.get("top_k") or DEFAULT_TOP_K)

        if not startup_name:
            raise ValueError("startup_name is required.")
        if not market_query:
            raise ValueError("market_query is required.")

        search_query = self._build_search_query(
            market_query=market_query,
            company_context=company_context,
        )
        report_categories = self._resolve_report_categories(market_query, company_context)
        evidence_rows = self._retrieve_market_evidence(
            search_query,
            report_categories=report_categories,
            top_k=top_k,
        )

        if not evidence_rows:
            evaluated = self._build_insufficient_evaluation(
                startup_name=startup_name,
                market_query=market_query,
            )
        else:
            evaluated = self._evaluate_with_llm(
                startup_name=startup_name,
                market_query=market_query,
                company_context=company_context,
                report_categories=report_categories,
                evidence_rows=evidence_rows,
            )

        return {
            "startup_name": startup_name,
            "market_query": market_query,
            "search_query": search_query,
            **evaluated,
        }

    def run(
        self,
        startup_name: str,
        market_query: str,
        company_context: str = "",
        top_k: int = DEFAULT_TOP_K,
    ) -> MarketEvaluationOutput:
        """단독 실행."""

        return self(
            {
                "startup_name": startup_name,
                "market_query": market_query,
                "company_context": company_context,
                "top_k": top_k,
            }
        )

    # 이하 내부 구현은 기존 market-evaluation-agent/agent.py와 동일하게 유지
    def _build_search_query(
        self,
        *,
        market_query: str,
        company_context: str,
    ) -> str:
        report_categories = self._resolve_report_categories(market_query, company_context)
        parts = report_categories + [
            market_query,
            company_context,
            "시장 규모 매출 사업체 수 성장률 도입",
        ]
        return " ".join(part.strip() for part in parts if part.strip())

    def _resolve_report_categories(self, market_query: str, company_context: str) -> List[str]:
        source = f"{market_query} {company_context}"
        categories: List[str] = []
        for keywords, mapped_categories in REPORT_CATEGORY_RULES:
            if any(keyword in source for keyword in keywords):
                categories.extend(mapped_categories)
        if not categories:
            categories.extend(["서비스용 로봇", "제조업용 로봇"])
        return self._unique_preserve_order(categories)

    def _get_vector_store(self) -> Any:
        """시장성 평가 전용 vector store를 지연 로드한다."""
        if self.vector_store is not None:
            return self.vector_store
        try:
            from vectorstore import AGENT_MARKET_EVAL, get_vector_store
        except ModuleNotFoundError as exc:
            missing_module = exc.name or "unknown module"
            raise RuntimeError(
                "시장성 평가 vector store를 불러오지 못했습니다. "
                f"누락 모듈: '{missing_module}'. "
                "`.venv`를 활성화한 뒤 `pip install -r requirements.txt`를 실행하세요."
            ) from exc
        self.vector_store = get_vector_store(AGENT_MARKET_EVAL)
        return self.vector_store

    def _retrieve_market_evidence(
        self,
        search_query: str,
        *,
        report_categories: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """market_eval vector store에서 유사 문서를 조회한다."""
        store = self._get_vector_store()
        docs_with_scores: List[Tuple[Any, Any]] = []
        fetch_k = max(top_k * FETCH_MULTIPLIER, 12)

        if hasattr(store, "similarity_search_with_score"):
            docs_with_scores = store.similarity_search_with_score(search_query, k=fetch_k)
        elif hasattr(store, "similarity_search"):
            docs = store.similarity_search(search_query, k=fetch_k)
            docs_with_scores = [(doc, None) for doc in docs]

        evidence_rows: List[Dict[str, Any]] = []
        for item in docs_with_scores:
            if isinstance(item, tuple) and len(item) >= 1:
                doc = item[0]
                score = item[1] if len(item) > 1 else None
            else:
                doc = item
                score = None
            content = getattr(doc, "page_content", "") or ""
            metadata = getattr(doc, "metadata", {}) or {}
            if not str(content).strip():
                continue
            evidence_rows.append(
                {
                    "content": str(content).strip(),
                    "metadata": metadata,
                    "score": self._score_to_similarity(score),
                }
            )
        return self._rerank_evidence_rows(
            evidence_rows,
            search_query=search_query,
            report_categories=report_categories,
            top_k=top_k,
        )

    def _evaluate_with_llm(
        self,
        *,
        startup_name: str,
        market_query: str,
        company_context: str,
        report_categories: List[str],
        evidence_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = MARKET_EVALUATION_PROMPT.format(
            startup_name=startup_name,
            market_query=market_query,
            company_context=f"{company_context or '정보 없음'} / 보고서 분류어: {', '.join(report_categories)}",
            market_context=self._format_evidence_rows(evidence_rows),
        )
        payload = self.client.complete_json(
            system_prompt=MARKET_EVALUATION_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        return self._normalize_output(payload, evidence_rows)

    def _build_insufficient_evaluation(
        self,
        *,
        startup_name: str,
        market_query: str,
    ) -> Dict[str, Any]:
        return {
            "market_summary": f"{startup_name}의 `{market_query}` 관련 시장 자료가 충분히 검색되지 않았다.",
            "market_size": "자료 부족",
            "growth_drivers": [],
            "competition_analysis": "자료 부족",
            "customer_adoption": "자료 부족",
            "key_risks": [
                "market_eval vector store에서 관련 시장 자료를 더 확보해야 한다.",
                "검색 질의를 보고서 분류어 기준으로 더 좁힐 필요가 있다.",
            ],
            "evidence": [],
        }

    def _normalize_output(
        self,
        payload: Dict[str, Any],
        evidence_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence = self._as_list(payload.get("evidence"))
        if not evidence:
            evidence = self._format_evidence_list(evidence_rows)
        return {
            "market_summary": self._as_string(payload.get("market_summary"), "시장성 요약 정보 없음"),
            "market_size": self._as_string(payload.get("market_size"), "시장 규모 정보 없음"),
            "growth_drivers": self._as_list(payload.get("growth_drivers")),
            "target_industries": self._as_list(payload.get("target_industries")),
            "competition_analysis": self._as_string(
                payload.get("competition_analysis"),
                "경쟁 환경 정보 없음",
            ),
            "customer_adoption": self._as_string(
                payload.get("customer_adoption"),
                "고객 도입 가능성 정보 없음",
            ),
            "key_risks": self._as_list(payload.get("key_risks")),
            "evidence": evidence,
        }

    def _format_evidence_rows(self, evidence_rows: List[Dict[str, Any]]) -> str:
        if not evidence_rows:
            return "검색 결과 없음"
        formatted: List[str] = []
        for row in evidence_rows:
            metadata = row.get("metadata", {}) or {}
            source = metadata.get("source") or metadata.get("source_file") or metadata.get("filename") or "unknown"
            page = metadata.get("page")
            location = f"{source} p.{page}" if page is not None else str(source)
            content = row.get("content", "")
            preview = content[:400] + "..." if len(content) > 400 else content
            formatted.append(f"- [{location}] {preview}")
        return "\n".join(formatted)

    def _format_evidence_list(self, evidence_rows: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        for row in evidence_rows[:5]:
            metadata = row.get("metadata", {}) or {}
            source = metadata.get("source_file") or metadata.get("source") or metadata.get("filename") or "unknown"
            page = metadata.get("page")
            location = f"{source} p.{page}" if page is not None else str(source)
            preview = self._extract_best_snippet(row.get("content", ""))
            evidence.append(f"{location}: {preview}")
        return evidence

    def _as_string(self, value: Any, default: str = "") -> str:
        return value.strip() if isinstance(value, str) and value.strip() else default

    def _as_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _score_to_similarity(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0.0 <= numeric <= 1.0:
                return 1.0 - numeric
            return 1.0 / (1.0 + max(numeric, 0.0))
        return 0.0

    def _extract_query_keywords(self, search_query: str) -> List[str]:
        stopwords = {
            "국내",
            "시장",
            "규모",
            "성장성",
            "성장률",
            "자동화",
            "로봇",
            "기반",
            "검토",
            "가능성",
            "대상",
            "및",
        }
        tokens = [token.strip(",") for token in search_query.split()]
        filtered = [token for token in tokens if len(token) >= 2 and token not in stopwords]
        return self._unique_preserve_order(filtered)

    def _rerank_evidence_rows(
        self,
        evidence_rows: List[Dict[str, Any]],
        *,
        search_query: str,
        report_categories: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        query_keywords = self._extract_query_keywords(search_query)
        scored_rows: List[Tuple[float, Dict[str, Any]]] = []
        for row in evidence_rows:
            content = row.get("content", "")
            score = float(row.get("score") or 0.0)
            category_hits = sum(1 for category in report_categories if category in content)
            keyword_hits = sum(1 for keyword in query_keywords if keyword in content)
            penalty = 0.0
            if "업체의 주된 업종 기준이 아닌" in content:
                penalty += 0.2
            combined = score + (category_hits * 0.6) + (keyword_hits * 0.15) - penalty
            scored_rows.append((combined, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        results: List[Dict[str, Any]] = []
        seen_keys = set()
        for _, row in scored_rows:
            metadata = row.get("metadata", {}) or {}
            dedupe_key = (metadata.get("source_file"), metadata.get("page"), metadata.get("chunk_index"))
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            results.append(row)
            if len(results) >= top_k:
                break
        return results

    def _extract_best_snippet(self, content: str) -> str:
        lines = [self._normalize_line(line) for line in content.splitlines()]
        candidate_lines = [
            line
            for line in lines
            if line
            and any(keyword in line for keyword in ["매출", "생산액", "사업체", "성장", "수출", "수입", "비중", "도입"])
        ]
        source = candidate_lines[0] if candidate_lines else self._normalize_line(content)
        return source[:180] + "..." if len(source) > 180 else source

    def _normalize_line(self, text: str) -> str:
        return " ".join((text or "").split())

    def _unique_preserve_order(self, values: List[str]) -> List[str]:
        seen = set()
        results: List[str] = []
        for value in values:
            normalized = self._normalize_line(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results

