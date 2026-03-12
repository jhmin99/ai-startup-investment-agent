"""질문 재정의 에이전트."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from urllib import error, request

from query_refinement.prompts import build_user_prompt, get_system_prompt
from query_refinement.state import (
    QueryRefinementOutput,
    QueryRefinementState,
    validate_query_refinement_state,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _as_string(value: Any, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _load_env_file() -> None:
    """로컬 실행 시 `.env` 설정을 읽는다."""

    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(".env")
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


class QueryRefinementLLMClient(Protocol):
    """질문 재정의에 필요한 최소 LLM 인터페이스."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """모델이 생성한 JSON 객체를 반환한다."""


@dataclass
class OpenAIChatClient:
    """OpenAI 호환 Chat Completions API 클라이언트."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "OpenAIChatClient":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for AI-based query refinement. "
                "Set OPENAI_API_KEY and rerun."
            )
        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
        raw_response = self._post(url, payload)
        content = raw_response["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Model response content was not a JSON string.")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {content}") from exc

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
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


class QueryRefinementAgent:
    """초기 검색 실패 후 재검색 질의를 다시 쓰는 최소 에이전트."""

    def __init__(self, client: Optional[QueryRefinementLLMClient] = None) -> None:
        self.client = client or OpenAIChatClient.from_env()

    def refine(self, state: QueryRefinementState) -> QueryRefinementOutput:
        validate_query_refinement_state(state)

        payload = self.client.complete_json(
            system_prompt=get_system_prompt(),
            user_prompt=build_user_prompt(state),
        )
        return self._build_output(payload, state)

    def _build_output(
        self,
        payload: Dict[str, Any],
        state: QueryRefinementState,
    ) -> QueryRefinementOutput:
        raw_question = state["raw_question"]
        refined_query = _as_string(payload.get("refined_query"), default=raw_question)
        # 최소 버전에서는 대표 질의와 대체 질의 몇 개만 유지한다.
        reformulated_queries = _unique_preserve_order(
            [refined_query] + _as_list(payload.get("reformulated_queries")) + [raw_question]
        )[:3]

        if not reformulated_queries:
            reformulated_queries = [refined_query]

        retry_strategy = self._normalize_retry_strategy(
            payload.get("retry_strategy"),
            refined_query=refined_query,
            reformulated_queries=reformulated_queries,
        )

        return {
            "refined_query": refined_query,
            "reformulated_queries": reformulated_queries,
            "retry_strategy": retry_strategy,
        }

    def _normalize_retry_strategy(
        self,
        payload: Any,
        *,
        refined_query: str,
        reformulated_queries: List[str],
    ) -> str:
        if isinstance(payload, str) and payload.strip():
            return payload.strip()

        if isinstance(payload, dict):
            primary_query = _as_string(payload.get("primary_query"), default=refined_query)
            fallback_queries = _unique_preserve_order(
                _as_list(payload.get("fallback_queries")) + reformulated_queries[1:]
            )
            if fallback_queries:
                return (
                    f"1차는 `{primary_query}` 로 재검색한다. "
                    f"결과가 부족하면 `{fallback_queries[0]}` 부터 순차 적용한다."
                )
            return f"1차는 `{primary_query}` 로 재검색한다."

        if len(reformulated_queries) > 1:
            return (
                f"1차는 `{refined_query}` 로 재검색한다. "
                f"결과가 부족하면 `{reformulated_queries[1]}` 부터 순차 적용한다."
            )
        return f"1차는 `{refined_query}` 로 재검색한다."
