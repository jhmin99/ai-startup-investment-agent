from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    # 기본 모델은 실제 OpenAI 퍼블릭 엔드포인트에서 바로 쓸 수 있는 모델로 설정
    # (env에 OPENAI_MODEL이 지정되어 있으면 그 값을 우선 사용)
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature: float = 0.0


def get_chat_model() -> ChatOpenAI:
    settings = Settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )

