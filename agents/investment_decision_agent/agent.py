from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from .schemas import InvestmentDecision, LLMScorecard, ScoreItem, WeightedScore
from .utils import get_chat_model


WEIGHTS: Dict[str, int] = {
    "market": 35,
    "technology": 25,
    "competitiveness": 20,
    "traction": 20,
}


def _fallback_score(text: str, positive_keywords: List[str]) -> int:
    normalized = (text or "").lower()
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


def _build_fallback_scorecard(state: Dict[str, Any]) -> LLMScorecard:
    """
    LLM 호출이 실패했을 때를 위한 한국어/구조화 필드 기반 휴리스틱 점수.

    - 영어 키워드 매칭 대신, 지금 파이프라인에서 실제로 채워지는 필드들을 사용해서
      회사마다 점수가 다르게 나오도록 설계한다.
    """
    tech = state.get("technology_summary") or {}
    market = state.get("market_assessment") or {}
    exploration = state.get("exploration_summary") or {}
    profile = state.get("startup_profile") or {}

    strengths = list(tech.get("strengths") or [])
    differentiation = str(tech.get("differentiation") or "").strip()

    market_size = str(market.get("market_size") or "").strip()
    growth_drivers = list(market.get("growth_drivers") or [])
    target_industries = list(market.get("target_industries") or [])

    competitor_summary = str(exploration.get("competitor_summary") or "").strip()
    traction_summary = str(exploration.get("traction_summary") or "").strip()

    customers = list(profile.get("customers") or [])
    fundraising_history = list(profile.get("fundraising_history") or [])
    patents = list(profile.get("patents") or [])

    # --- 시장 점수: 시장 정보가 얼마나 있는지 ---
    market_score = 1
    if market_size and market_size != "정보 부족":
        market_score += 1
    if growth_drivers:
        market_score += 1
    if target_industries:
        market_score += 1
    # 시장 관련 서술이 traction/competitor에 조금이라도 있으면 +1
    if any(
        kw in (market_size + traction_summary + competitor_summary)
        for kw in ["시장", "성장", "수요", "로봇", "자동화"]
    ):
        market_score += 1
    market_score = max(1, min(5, market_score))

    # --- 기술 점수: 강점/차별성/특허 등 ---
    technology_score = 1
    if strengths:
        technology_score += 1
    if len(strengths) >= 3:
        technology_score += 1
    if differentiation and differentiation != "정보 없음":
        technology_score += 1
    if patents:
        technology_score += 1
    technology_score = max(1, min(5, technology_score))

    # --- 경쟁력 점수: 경쟁사 대비 설명 여부 ---
    competitiveness_score = 1
    if differentiation and "정보 부족" not in differentiation:
        competitiveness_score += 1
    if competitor_summary:
        competitiveness_score += 1
    if any(kw in competitor_summary for kw in ["차별", "우위", "경쟁", "비교"]):
        competitiveness_score += 1
    competitiveness_score = max(1, min(5, competitiveness_score))

    # --- 실적 점수: traction/고객/투자 이력 ---
    traction_score = 1
    if customers:
        traction_score += 1
    if fundraising_history:
        traction_score += 1
    if traction_summary:
        traction_score += 1
    if any(kw in traction_summary for kw in ["고객", "파일럿", "도입", "매출", "계약", "PoC"]):
        traction_score += 1
    traction_score = max(1, min(5, traction_score))

    return LLMScorecard(
        market=ScoreItem(score=market_score, rationale="LLM 실패로 구조화 필드 기반 휴리스틱 점수 사용"),
        technology=ScoreItem(
            score=technology_score, rationale="LLM 실패로 구조화 필드 기반 휴리스틱 점수 사용"
        ),
        competitiveness=ScoreItem(
            score=competitiveness_score, rationale="LLM 실패로 구조화 필드 기반 휴리스틱 점수 사용"
        ),
        traction=ScoreItem(
            score=traction_score, rationale="LLM 실패로 구조화 필드 기반 휴리스틱 점수 사용"
        ),
    )


def _score_with_llm(state: Dict[str, Any]) -> LLMScorecard:
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

    # state가 pydantic을 쓰든 dict를 쓰든 최대한 그대로 문자열화해서 전달
    def dump_json(v: Any) -> str:
        if hasattr(v, "model_dump_json"):
            return v.model_dump_json(indent=2, ensure_ascii=False)
        return str(v)

    return (prompt | llm).invoke(
        {
            "user_query": state.get("user_query", ""),
            "startup_profile": dump_json(state.get("startup_profile")),
            "exploration_summary": dump_json(state.get("exploration_summary")),
            "technology_summary": dump_json(state.get("technology_summary")),
            "market_assessment": dump_json(state.get("market_assessment")),
        }
    )


def _calculate_weighted_score(scorecard: LLMScorecard) -> WeightedScore:
    market_weighted = scorecard.market.score / 5 * WEIGHTS["market"]
    technology_weighted = scorecard.technology.score / 5 * WEIGHTS["technology"]
    competitiveness_weighted = scorecard.competitiveness.score / 5 * WEIGHTS["competitiveness"]
    traction_weighted = scorecard.traction.score / 5 * WEIGHTS["traction"]

    total = market_weighted + technology_weighted + competitiveness_weighted + traction_weighted

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


def _collect_missing_information(state: Dict[str, Any]) -> List[str]:
    missing: List[str] = []

    profile = state.get("startup_profile") or {}
    market = state.get("market_assessment") or {}

    def _as_list(v: Any) -> List[str]:
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str)]
        return []

    if not _as_list(profile.get("customers")):
        missing.append("고객사 또는 파일럿 도입 정보")
    if not _as_list(profile.get("fundraising_history")):
        missing.append("투자 유치 이력")
    if not _as_list(profile.get("patents")):
        missing.append("특허 또는 독자 기술 증빙")
    if not (isinstance(market.get("commercialization_risk"), str) and market.get("commercialization_risk").strip()):
        missing.append("상용화 리스크 정리")

    return missing


def investment_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node 스타일(입력 state dict -> state dict).

    반환:
    - investment_decision: InvestmentDecision (pydantic)
    """
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
        verdict=verdict,  # type: ignore[arg-type]
        requires_web_search=requires_web_search,
        decision_rationale=rationale,
        missing_information=missing_information,
    )

    state["investment_decision"] = decision
    return state


class InvestmentDecisionAgent:
    """
    investment_decision_node를 class로 감싼 버전.
    """

    def run(self, state: Dict[str, Any]) -> InvestmentDecision:
        out = investment_decision_node(state)
        decision = out.get("investment_decision")
        if isinstance(decision, InvestmentDecision):
            return decision
        if hasattr(decision, "model_dump"):
            # pydantic 호환
            return InvestmentDecision.model_validate(decision)
        raise ValueError("investment_decision was not produced.")

