from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Reference(BaseModel):
    title: str = Field(description="출처 제목")
    url: str = Field(description="출처 URL")
    source_type: Literal["rag", "web", "manual"] = Field(default="rag")


class FinalReport(BaseModel):
    html: str

