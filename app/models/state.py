from __future__ import annotations

from typing import Annotated, List, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from app.models.schemas import (
    ExplorationSummary,
    FinalReport,
    InvestmentDecision,
    MarketAssessment,
    StartupProfile,
    TechnologySummary,
)


class InvestmentGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    startup_profile: StartupProfile
    exploration_summary: ExplorationSummary
    technology_summary: TechnologySummary
    market_assessment: MarketAssessment
    investment_decision: NotRequired[InvestmentDecision]
    final_report: NotRequired[FinalReport]
