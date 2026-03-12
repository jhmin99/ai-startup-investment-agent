from app.agents.investment_decision_agent import _calculate_weighted_score
from app.models.schemas import LLMScorecard, ScoreItem


def test_weighted_score() -> None:
    scorecard = LLMScorecard(
        market=ScoreItem(score=5, rationale=""),
        technology=ScoreItem(score=4, rationale=""),
        competitiveness=ScoreItem(score=3, rationale=""),
        traction=ScoreItem(score=3, rationale=""),
    )
    weighted = _calculate_weighted_score(scorecard)
    assert weighted.total == 80.0
