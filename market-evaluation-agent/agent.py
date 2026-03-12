"""
시장성 평가 에이전트 - 단일 노드로 캡슐화

상위 그래프에서 사용:
    market_eval_agent = MarketEvaluationAgent()
    workflow.add_node("market_evaluation", market_eval_agent)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple
from urllib import error, request

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        """python-dotenv가 없어도 실행은 가능하게 둔다."""

        return False

from prompts import MARKET_EVALUATION_PROMPT
from state import MarketEvaluationOutput

load_dotenv()

# `python main.py ...` 형태로 실행할 때 프로젝트 루트를 import path에 추가한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 상수 정의
# ============================================================


DEFAULT_MODEL = "gpt-4o"
DEFAULT_TOP_K = 4
FETCH_MULTIPLIER = 4
DEFAULT_TIMEOUT_SECONDS = 30.0

REPORT_CATEGORY_RULES = [
    (
        ["스마트팩토리", "제조", "생산라인", "공정", "전장", "전자"],
        ["제조업용 로봇", "로봇시스템"],
    ),
    (
        ["건설", "양중", "도장", "프린팅"],
        ["전문서비스용 로봇", "서비스용 로봇"],
    ),
    (
        ["물류", "배송", "병원", "호텔", "리테일", "자율주행"],
        ["전문서비스용 로봇", "서비스용 로봇", "로봇서비스"],
    ),
    (
        ["개인", "가정", "돌봄", "청소"],
        ["개인서비스용 로봇", "서비스용 로봇"],
    ),
]

MARKET_EVALUATION_SYSTEM_PROMPT = """너는 Robotics 스타트업 투자 검토를 위한 시장성 평가 분석가다.
반드시 검색 근거만 바탕으로 평가하고, JSON object만 반환한다.
근거가 약하면 과장하지 말고 부족하다고 명시한다.
"""


# ============================================================
# 시장성 평가 에이전트
# ============================================================


class MarketEvaluationAgent:
    """
    시장 자료 기반으로 스타트업의 시장성을 평가하는 에이전트.

    - 검색은 `vectorstore.py`의 market_eval 스토어만 사용한다.
    - 결과는 투자 판단 단계로 넘길 구조화된 시장성 요약만 반환한다.
    """

    # --------------------------------------------------------
    # 초기화
    # --------------------------------------------------------

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        vector_store: Optional[Any] = None,
        client: Optional["MarketEvaluationLLMClient"] = None,
    ) -> None:
        self.model = model
        self.vector_store = vector_store
        self.client = client or OpenAIChatClient.from_env(
            openai_api_key=openai_api_key,
            default_model=model,
        )

    # --------------------------------------------------------
    # 공개 인터페이스
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 검색 질의 생성
    # --------------------------------------------------------

    def _build_search_query(
        self,
        *,
        market_query: str,
        company_context: str,
    ) -> str:
        """시장 자료 검색용 질의를 단순 조합한다."""

        report_categories = self._resolve_report_categories(market_query, company_context)
        parts = report_categories + [
            market_query,
            company_context,
            "시장 규모 매출 사업체 수 성장률 도입",
        ]
        return " ".join(part.strip() for part in parts if part.strip())

    def _resolve_report_categories(self, market_query: str, company_context: str) -> List[str]:
        """시장 보고서 분류어로 검색 질의를 보정한다."""

        source = f"{market_query} {company_context}"
        categories: List[str] = []

        for keywords, mapped_categories in REPORT_CATEGORY_RULES:
            if any(keyword in source for keyword in keywords):
                categories.extend(mapped_categories)

        if not categories:
            categories.extend(["서비스용 로봇", "제조업용 로봇"])

        return self._unique_preserve_order(categories)

    # --------------------------------------------------------
    # vectorstore 검색
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 시장성 평가
    # --------------------------------------------------------

    def _evaluate_with_llm(
        self,
        *,
        startup_name: str,
        market_query: str,
        company_context: str,
        report_categories: List[str],
        evidence_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """검색된 시장 자료를 LLM으로 구조화 평가한다."""

        prompt = MARKET_EVALUATION_PROMPT.format(
            startup_name=startup_name,
            market_query=market_query,
            company_context=(
                f"{company_context or '정보 없음'} / 보고서 분류어: {', '.join(report_categories)}"
            ),
            market_context=self._format_evidence_rows(evidence_rows),
        )

        try:
            payload = self.client.complete_json(
                system_prompt=MARKET_EVALUATION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
        except ValueError as exc:
            raise RuntimeError(f"시장성 평가 LLM 호출에 실패했습니다: {exc}") from exc

        return self._normalize_output(payload, evidence_rows)

    def _build_insufficient_evaluation(
        self,
        *,
        startup_name: str,
        market_query: str,
    ) -> Dict[str, Any]:
        """검색 근거가 부족할 때 최소 결과를 반환한다."""

        return {
            "market_summary": (
                f"{startup_name}의 `{market_query}` 관련 시장 자료가 충분히 검색되지 않았다."
            ),
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
        """LLM 응답을 최소 출력 계약으로 정규화한다."""

        evidence = self._as_list(payload.get("evidence"))
        if not evidence:
            evidence = self._format_evidence_list(evidence_rows)

        return {
            "market_summary": self._as_string(payload.get("market_summary"), "시장성 요약 정보 없음"),
            "market_size": self._as_string(payload.get("market_size"), "시장 규모 정보 없음"),
            "growth_drivers": self._as_list(payload.get("growth_drivers")),
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

    # --------------------------------------------------------
    # 포맷 유틸
    # --------------------------------------------------------

    def _format_evidence_rows(self, evidence_rows: List[Dict[str, Any]]) -> str:
        """프롬프트용 시장 자료 문자열을 만든다."""

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
        """투자 판단 단계로 넘길 짧은 evidence 리스트를 만든다."""

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
        """문자열 필드를 안전하게 정규화한다."""

        return value.strip() if isinstance(value, str) and value.strip() else default

    def _as_list(self, value: Any) -> List[str]:
        """리스트 필드를 안전하게 정규화한다."""

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _score_to_similarity(self, value: Any) -> float:
        """distance 또는 score를 0~1 유사도로 바꾼다."""

        if isinstance(value, (int, float)):
            numeric = float(value)
            if 0.0 <= numeric <= 1.0:
                return 1.0 - numeric
            return 1.0 / (1.0 + max(numeric, 0.0))
        return 0.0

    def _extract_query_keywords(self, search_query: str) -> List[str]:
        """검색 query에서 의미 있는 키워드를 추출한다."""

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
        """벡터 검색 결과를 도메인 키워드 기준으로 한 번 더 정렬한다."""

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
        """근거로 보여줄 짧은 문장을 뽑는다."""

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
        """공백과 줄바꿈이 깨진 문장을 보기 좋게 정리한다."""

        return " ".join((text or "").split())

    def _unique_preserve_order(self, values: List[str]) -> List[str]:
        """중복을 제거하되 기존 순서를 유지한다."""

        seen = set()
        results: List[str] = []
        for value in values:
            normalized = self._normalize_line(value)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
        return results


def _load_env_file() -> None:
    """로컬 실행 시 `.env` 설정을 읽는다."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


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
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        response_payload = self._post(url, payload)
        content = response_payload["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model response content was not a JSON string.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {content}") from exc

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ValueError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise ValueError(f"LLM request failed: {exc.reason}") from exc
