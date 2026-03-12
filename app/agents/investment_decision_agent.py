from __future__ import annotations

from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

from app.core.llm import get_chat_model
from app.models.schemas import (
    InvestmentDecision,
    LLMScorecard,
    ScoreItem,
    WeightedScore,
)
from app.models.state import InvestmentGraphState


WEIGHTS: Dict[str, int] = {
    "market": 35,
    "technology": 25,
    "competitiveness": 20,
    "traction": 20,
}


def _fallback_score(text: str, positive_keywords: List[str]) -> int:
    normalized = text.lower()
    hit_count = sum(1 for keyword in positive_keywords if keyword.lower() in normalized)

    if hit_count >= 4:
        return 5
    if hit_count == 3:
        return 4
    if hit_count == 2:
        return 3
    if hit_count == 1:
        return 2
    return 1


def _build_fallback_scorecard(state: InvestmentGraphState) -> LLMScorecard:
    tech = state["technology_summary"]
    market = state["market_assessment"]
    exploration = state["exploration_summary"]
    profile = state["startup_profile"]

    market_score = _fallback_score(
        " ".join(
            [
                market.market_size,
                market.scalability,
                *market.growth_drivers,
                *market.target_industries,
            ]
        ),
        [
            "growth",
            "expansion",
            "automation",
            "humanoid",
            "factory",
            "logistics",
            "service robot",
            "high demand",
        ],
    )

    technology_score = _fallback_score(
        " ".join(
            [
                tech.core_technology,
                tech.differentiation,
                *tech.strengths,
                *profile.patents,
            ]
        ),
        [
            "patent",
            "proprietary",
            "vision",
            "sensor fusion",
            "foundation model",
            "autonomy",
            "precision",
            "multimodal",
        ],
    )

    competitiveness_score = _fallback_score(
        exploration.competitor_summary,
        [
            "differentiated",
            "cost advantage",
            "speed",
            "accuracy",
            "integration",
            "deployment",
            "partnership",
            "moat",
        ],
    )

    traction_score = _fallback_score(
        " ".join(
            [
                exploration.traction_summary,
                *profile.customers,
                *profile.fundraising_history,
            ]
        ),
        [
            "customer",
            "pilot",
            "revenue",
            "series",
            "contract",
            "deployment",
            "recurring",
            "po",
        ],
    )

    return LLMScorecard(
        market=ScoreItem(
            score=market_score, rationale="환경 변수 미설정으로 휴리스틱 점수 사용"
        ),
        technology=ScoreItem(
            score=technology_score, rationale="환경 변수 미설정으로 휴리스틱 점수 사용"
        ),
        competitiveness=ScoreItem(
            score=competitiveness_score,
            rationale="환경 변수 미설정으로 휴리스틱 점수 사용",
        ),
        traction=ScoreItem(
            score=traction_score, rationale="환경 변수 미설정으로 휴리스틱 점수 사용"
        ),
    )


def _score_with_llm(state: InvestmentGraphState) -> LLMScorecard:
    llm = get_chat_model().with_structured_output(LLMScorecard)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
당신은 로보틱스 스타트업 투자 심사역입니다.
주어진 정보만 사용해서 아래 4개 항목을 1~5점으로 채점하세요.

채점 원칙:
- 5점: 업계 상위권 수준의 강한 증거가 충분함
- 4점: 강한 편이나 일부 불확실성 존재
- 3점: 평균 수준, 장단점 혼재
- 2점: 약점이 뚜렷하거나 증거 부족
- 1점: 매우 부정적이거나 근거가 거의 없음

반드시 각 항목별 rationale에는 왜 그 점수를 줬는지 2~3문장으로 적으세요.
추정은 가능하지만, 없는 사실을 만들면 안 됩니다.
                """.strip(),
            ),
            (
                "human",
                """
[사용자 질의]
{user_query}

[기업 기본 정보]
{startup_profile}

[스타트업 탐색 결과]
{exploration_summary}

[기술 요약 결과]
{technology_summary}

[시장성 평가 결과]
{market_assessment}
                """.strip(),
            ),
        ]
    )

    return (prompt | llm).invoke(
        {
            "user_query": state["user_query"],
            "startup_profile": state["startup_profile"].model_dump_json(
                indent=2, ensure_ascii=False
            ),
            "exploration_summary": state["exploration_summary"].model_dump_json(
                indent=2, ensure_ascii=False
            ),
            "technology_summary": state["technology_summary"].model_dump_json(
                indent=2, ensure_ascii=False
            ),
            "market_assessment": state["market_assessment"].model_dump_json(
                indent=2, ensure_ascii=False
            ),
        }
    )


def _calculate_weighted_score(scorecard: LLMScorecard) -> WeightedScore:
    market_weighted = scorecard.market.score / 5 * WEIGHTS["market"]
    technology_weighted = scorecard.technology.score / 5 * WEIGHTS["technology"]
    competitiveness_weighted = (
        scorecard.competitiveness.score / 5 * WEIGHTS["competitiveness"]
    )
    traction_weighted = scorecard.traction.score / 5 * WEIGHTS["traction"]

    total = (
        market_weighted
        + technology_weighted
        + competitiveness_weighted
        + traction_weighted
    )

    return WeightedScore(
        market=round(market_weighted, 1),
        technology=round(technology_weighted, 1),
        competitiveness=round(competitiveness_weighted, 1),
        traction=round(traction_weighted, 1),
        total=round(total, 1),
    )


def _determine_verdict(total_score: float) -> str:
    if total_score >= 80:
        return "적극 투자"
    if total_score >= 70:
        return "투자 가능"
    if total_score < 50:
        return "투자 위험"
    return "보류"


def _collect_missing_information(state: InvestmentGraphState) -> List[str]:
    missing: List[str] = []

    profile = state["startup_profile"]
    market = state["market_assessment"]

    if not profile.customers:
        missing.append("고객사 또는 파일럿 도입 정보")
    if not profile.fundraising_history:
        missing.append("투자 유치 이력")
    if not profile.patents:
        missing.append("특허 또는 독자 기술 증빙")
    if not market.commercialization_risk:
        missing.append("상용화 리스크 정리")

    return missing


def investment_decision_node(state: InvestmentGraphState) -> InvestmentGraphState:
    try:
        scorecard = _score_with_llm(state)
    except Exception:
        scorecard = _build_fallback_scorecard(state)

    weighted = _calculate_weighted_score(scorecard)
    verdict = _determine_verdict(weighted.total)
    missing_information = _collect_missing_information(state)
    requires_web_search = weighted.total < 70 or len(missing_information) >= 2

    rationale = [
        f"시장성 {scorecard.market.score}/5: {scorecard.market.rationale}",
        f"기술력 {scorecard.technology.score}/5: {scorecard.technology.rationale}",
        f"경쟁력 {scorecard.competitiveness.score}/5: {scorecard.competitiveness.rationale}",
        f"실적 {scorecard.traction.score}/5: {scorecard.traction.rationale}",
    ]

    decision = InvestmentDecision(
        scorecard=scorecard,
        weighted_score=weighted,
        verdict=verdict,
        requires_web_search=requires_web_search,
        decision_rationale=rationale,
        missing_information=missing_information,
    )

    state["investment_decision"] = decision
    return state
