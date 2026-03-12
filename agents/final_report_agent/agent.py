from __future__ import annotations

from html import escape
from itertools import chain
from typing import Any, Dict, List

from .schemas import FinalReport, Reference
from .utils import (
    join_list,
    pass_fail_class,
    pass_fail_label,
    render_bullet_list,
    render_limit_chips,
    render_references,
    render_tag_list,
    safe_text,
    score_to_20,
    score_to_25,
    score_to_30,
    verdict_label,
    html_to_pdf,
)


def _collect_unique_references(state: Dict[str, Any]) -> list[Reference]:
    exploration_refs = _safe_list(_safe_get(state, ["exploration_summary", "references"], default=[]))
    technology_refs = _safe_list(_safe_get(state, ["technology_summary", "references"], default=[]))
    market_refs = _safe_list(_safe_get(state, ["market_assessment", "references"], default=[]))

    refs_raw = list(chain(exploration_refs, technology_refs, market_refs))

    unique_refs: list[Reference] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs_raw:
        try:
            r = ref if isinstance(ref, Reference) else Reference.model_validate(ref)
        except Exception:
            continue
        key = (r.title, r.url)
        if key not in seen:
            unique_refs.append(r)
            seen.add(key)
    return unique_refs


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_list(v: Any) -> list:
    return v if isinstance(v, list) else []


def _extract_strength_tags(state: Dict[str, Any]) -> list[str]:
    tech = state.get("technology_summary") or {}
    strengths = list(tech.get("strengths") or [])
    diff = tech.get("differentiation")
    if isinstance(diff, str) and diff.strip():
        strengths.append(diff.strip())
    return [s for s in strengths if isinstance(s, str) and s.strip()][:3]


def _extract_risk_tags(state: Dict[str, Any]) -> list[str]:
    tech = state.get("technology_summary") or {}
    market = state.get("market_assessment") or {}

    decision_raw = state.get("investment_decision")
    if decision_raw is None:
        decision = {}
    elif hasattr(decision_raw, "model_dump"):
        decision = decision_raw.model_dump()
    else:
        decision = decision_raw or {}

    risks: list[str] = []
    risks.extend([x for x in (tech.get("limitations") or []) if isinstance(x, str)])
    cr = market.get("commercialization_risk")
    if isinstance(cr, str) and cr.strip():
        risks.append(cr.strip())
    risks.extend([x for x in (decision.get("missing_information") or []) if isinstance(x, str)])

    deduped: list[str] = []
    seen = set()
    for risk in risks:
        normalized = risk.strip()
        if normalized and normalized not in seen:
            deduped.append(normalized)
            seen.add(normalized)
    return deduped[:3]


def _verdict_theme(verdict: str) -> dict[str, str]:
    if verdict == "적극 투자":
        return {"score_class": "score-blue", "tag_class": "tag-blue", "verdict_text_class": "verdict-blue"}
    if verdict == "투자 가능":
        return {"score_class": "score-green", "tag_class": "tag-green", "verdict_text_class": "verdict-green"}
    if verdict == "보류":
        return {"score_class": "score-orange", "tag_class": "tag-orange", "verdict_text_class": "verdict-orange"}
    return {"score_class": "score-danger", "tag_class": "tag-danger", "verdict_text_class": "verdict-danger"}


def _build_risk_items(state: Dict[str, Any]) -> str:
    tech = state.get("technology_summary") or {}
    market = state.get("market_assessment") or {}

    decision_raw = state.get("investment_decision")
    if decision_raw is None:
        decision = {}
    elif hasattr(decision_raw, "model_dump"):
        decision = decision_raw.model_dump()
    else:
        decision = decision_raw or {}

    limitations = [x for x in (tech.get("limitations") or []) if isinstance(x, str) and x.strip()]
    missing = [x for x in (decision.get("missing_information") or []) if isinstance(x, str) and x.strip()]

    risk_candidates = [
        ("🔧", "기술 리스크", limitations[0] if limitations else "기술 상용화 및 현장 적용 리스크가 존재합니다."),
        ("📋", "사업화 리스크", market.get("commercialization_risk") or "사업화 속도와 고객 전환까지 시간이 소요될 수 있습니다."),
        ("💸", "근거 부족", missing[0] if missing else "일부 투자 판단 근거가 충분하지 않습니다."),
        ("⚔️", "경쟁 리스크", (state.get("exploration_summary") or {}).get("competitor_summary") or "경쟁사 대비 우위 검증이 더 필요합니다."),
    ]

    items = []
    for icon, title, text in risk_candidates[:4]:
        items.append(
            f"""
            <div class="risk-item">
              <div class="r-icon">{icon}</div>
              <div class="r-text">
                <div class="r-title">{escape(title)}</div>
                {escape(str(text))}
              </div>
            </div>
            """.strip()
        )
    return "\n".join(items)


def _build_competitor_table(state: Dict[str, Any]) -> str:
    profile = state.get("startup_profile") or {}
    tech = state.get("technology_summary") or {}
    exploration = state.get("exploration_summary") or {}

    target_company = str(profile.get("company_name", ""))
    target_tech = str(tech.get("core_technology", ""))
    target_investment = join_list(profile.get("fundraising_history") or [], default="공개 정보 부족")
    target_customers = join_list(profile.get("customers") or [], default="공개 정보 부족")
    target_diff = str(tech.get("differentiation") or "차별점 정보 부족")

    competitor_a_name = "경쟁사 A"
    competitor_b_name = "경쟁사 B"

    generic_comp_a = "범용 자동화 솔루션"
    generic_comp_b = "대기업/글로벌 로봇 기업"
    generic_summary = str(exploration.get("competitor_summary") or "세부 경쟁 정보 부족")

    return f"""
    <table class="comp-table">
      <thead>
        <tr>
          <th>항목</th>
          <th class="highlight-col">{escape(target_company)}</th>
          <th>{escape(competitor_a_name)}</th>
          <th>{escape(competitor_b_name)}</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>핵심 기술</td>
          <td class="highlight-col">{escape(target_tech)}</td>
          <td>{escape(generic_comp_a)}</td>
          <td>{escape(generic_comp_b)}</td>
        </tr>
        <tr>
          <td>투자 규모</td>
          <td class="highlight-col">{escape(target_investment)}</td>
          <td>공개 정보 부족</td>
          <td>공개 정보 부족</td>
        </tr>
        <tr>
          <td>주요 고객</td>
          <td class="highlight-col">{escape(target_customers)}</td>
          <td>제조/물류 일반 고객군</td>
          <td>대기업 고객군</td>
        </tr>
        <tr>
          <td>차별점</td>
          <td class="highlight-col">✅ {escape(target_diff)}</td>
          <td>{escape(generic_summary[:60] + ("..." if len(generic_summary) > 60 else ""))}</td>
          <td>브랜드/자본 우위 가능성</td>
        </tr>
      </tbody>
    </table>
    """.strip()


def _build_company_report_section(state: Dict[str, Any], section_index: int | None = None) -> str:
    profile = state.get("startup_profile") or {}
    tech = state.get("technology_summary") or {}
    market = state.get("market_assessment") or {}

    # investment_decision은 pydantic 모델 또는 dict일 수 있다.
    decision_raw = state.get("investment_decision")
    if decision_raw is None:
        decision = {}
    elif hasattr(decision_raw, "model_dump"):
        # InvestmentDecision(pydantic) 호환
        decision = decision_raw.model_dump()
    else:
        decision = decision_raw

    references = _collect_unique_references(state)

    weighted = decision.get("weighted_score") or {}
    total_score = int(round(float(weighted.get("total", 0.0))))
    verdict_raw = str(decision.get("verdict") or "")
    verdict = verdict_label(verdict_raw)
    theme = _verdict_theme(verdict_raw)

    strength_tags = _extract_strength_tags(state)
    risk_tags = _extract_risk_tags(state)

    scorecard = decision.get("scorecard") or {}
    market_condition = (scorecard.get("market") or {}).get("score", 0) >= 3
    technology_condition = (scorecard.get("technology") or {}).get("score", 0) >= 3
    team_condition = (scorecard.get("competitiveness") or {}).get("score", 0) >= 3
    finance_condition = (scorecard.get("traction") or {}).get("score", 0) >= 3

    score_technology = score_to_30(int((scorecard.get("technology") or {}).get("score", 0) or 0))
    score_market = score_to_25(int((scorecard.get("market") or {}).get("score", 0) or 0))
    score_competitiveness = score_to_25(int((scorecard.get("competitiveness") or {}).get("score", 0) or 0))
    score_traction = score_to_20(int((scorecard.get("traction") or {}).get("score", 0) or 0))

    # 한계점(리밋) 영역: 기본 공통 한계 + 회사별 부족 정보 반영
    # (여기서는 high-level 한계만 표현하고, 세부 약점/리스크 문장은 tech 쪽에 남긴다.)
    limit_items: list[str] = ["공개 데이터만 사용", "시장 상황 변동 가능"]
    # 투자 판단에서 수집한 missing_information을 그대로 일부 노출 (필요 최소한만)
    for mi in decision.get("missing_information") or []:
        if isinstance(mi, str) and mi.strip():
            limit_items.append(mi.strip())
    # 고객/투자/특허 등 핵심 근거가 없으면 명시적으로 한계로 표기
    if not profile.get("customers"):
        limit_items.append("고객사/도입 실적 정보 부족")
    if not profile.get("fundraising_history"):
        limit_items.append("투자 유치 이력 정보 부족")
    if not profile.get("patents"):
        limit_items.append("특허/IP 근거 부족")
    if decision.get("requires_web_search"):
        limit_items.append("최신 웹 검색 보완 필요")

    # 중복 제거(순서 유지)
    seen_limits: set[str] = set()
    dedup_limits: list[str] = []
    for item in limit_items:
        if item not in seen_limits:
            seen_limits.add(item)
            dedup_limits.append(item)
    limit_items = dedup_limits

    tech_advantages = list(tech.get("strengths") or [])
    if isinstance(tech.get("differentiation"), str) and tech.get("differentiation").strip():
        tech_advantages.append(tech.get("differentiation").strip())

    section_title = f"{section_index}. {profile.get('company_name')}" if section_index else str(profile.get("company_name"))

    return f"""
<section class="company-section" id="company-{section_index or 1}">
  <div class="report-header">
    <div>
      <h1>🤖 {safe_text(section_title)}</h1>
      <p>로봇 스타트업 투자 분석 보고서 &nbsp;·&nbsp; 2026년 3월</p>
    </div>
    <div class="header-badge">
      <div class="score {theme['score_class']}">{total_score}<span style="font-size:1.2rem">/100</span></div>
      <div class="score-label">종합 투자 점수</div>
      <div class="verdict-tag {theme['tag_class']}">{escape(verdict)}</div>
    </div>
  </div>

  <div class="summary-strip">
    <div class="strip-card">
      <div class="label">핵심 사업</div>
      <div class="value">{safe_text(profile.get("one_liner"))}</div>
    </div>
    <div class="strip-card">
      <div class="label">핵심 장점</div>
      <div class="tag-list">
        {render_tag_list(strength_tags, "green")}
      </div>
    </div>
    <div class="strip-card">
      <div class="label">위험 요소</div>
      <div class="tag-list">
        {render_tag_list(risk_tags, "red")}
      </div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card">
      <div class="card-title"><div class="icon blue">🏢</div> 기업 소개</div>
      <div class="info-row"><span class="key">설립 연도</span><span class="val">{safe_text(str(profile.get("founded_year")) if profile.get("founded_year") else None)}</span></div>
      <div class="info-row"><span class="key">본사 위치</span><span class="val">{safe_text(profile.get("headquarters"))}</span></div>
      <div class="info-row"><span class="key">대표자</span><span class="val">{safe_text(profile.get("representative"))}</span></div>
      <div class="info-row"><span class="key">투자 유치</span><span class="val">{join_list(profile.get("fundraising_history") or [], default="공개 정보 부족")}</span></div>
      <div class="info-row"><span class="key">해결 문제</span><span class="val">{safe_text(profile.get("one_liner"))}</span></div>
    </div>

    <div class="card span2">
      <div class="card-title"><div class="icon purple">⚙️</div> 기술 분석</div>
      <div class="tech-grid">
        <div class="tech-item"><div class="t-label">로봇 기술 구조</div><div class="t-val">{safe_text(tech.get("technical_maturity"), default="모듈형/현장 최적화 구조")}</div></div>
        <div class="tech-item"><div class="t-label">AI 기술</div><div class="t-val">{safe_text(tech.get("core_technology"))}</div></div>
        <div class="tech-item"><div class="t-label">센서 기술</div><div class="t-val">{join_list(profile.get("products") or [], default="공개 정보 부족")}</div></div>
        <div class="tech-item"><div class="t-label">특허</div><div class="t-val">{join_list(profile.get("patents") or [], default="공개 정보 부족")}</div></div>
      </div>
      <div style="margin-top:14px;">
        <div class="card-title" style="margin-bottom:10px; border-bottom:none; padding-bottom:0;">경쟁사 대비 기술 장점</div>
        {render_bullet_list([x for x in tech_advantages if isinstance(x, str)], default_message="경쟁사 대비 기술 장점 정보가 부족합니다.")}
      </div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card span2">
      <div class="card-title"><div class="icon green">📊</div> 시장 분석</div>
      <div class="market-stats">
        <div class="m-stat"><div class="m-num">{safe_text(market.get("market_size"))}</div><div class="m-unit">시장 규모/설명</div></div>
        <div class="m-stat"><div class="m-num">{safe_text(join_list(market.get("growth_drivers") or [], default="성장 동인 정보 부족"))}</div><div class="m-unit">주요 성장 동인</div></div>
        <div class="m-stat"><div class="m-num">{safe_text(join_list(market.get("target_industries") or [], default="산업 정보 부족"))}</div><div class="m-unit">주요 수요 산업</div></div>
      </div>
      <ul class="bullet-list">
        <li>주요 산업: {escape(join_list(market.get("target_industries") or [], default="정보 부족"))}</li>
        <li>시장 성장 동인: {escape(join_list(market.get("growth_drivers") or [], default="정보 부족"))}</li>
        <li>시장 위험 요소: {escape(safe_text(market.get("commercialization_risk"), default="상용화 및 확장 과정의 불확실성 존재"))}</li>
      </ul>
    </div>

    <div class="card">
      <div class="card-title"><div class="icon red">⚠️</div> 위험 요소</div>
      <div class="risk-list">
        {_build_risk_items(state)}
      </div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card span2">
      <div class="card-title"><div class="icon orange">🏆</div> 경쟁 분석</div>
      {_build_competitor_table(state)}
    </div>

    <div class="card">
      <div class="card-title"><div class="icon teal">✅</div> 투자 평가</div>
      <div style="font-size:0.78rem; font-weight:700; color:#888; margin-bottom:8px;">기본 조건 평가</div>
      <div class="score-grid" style="margin-bottom:14px;">
        <div class="score-item"><span class="s-label">시장 규모</span><span class="score-pill {pass_fail_class(market_condition)}">{pass_fail_label(market_condition)}</span></div>
        <div class="score-item"><span class="s-label">기술 차별성</span><span class="score-pill {pass_fail_class(technology_condition)}">{pass_fail_label(technology_condition)}</span></div>
        <div class="score-item"><span class="s-label">팀/경쟁 역량</span><span class="score-pill {pass_fail_class(team_condition)}">{pass_fail_label(team_condition)}</span></div>
        <div class="score-item"><span class="s-label">실적/재무 근거</span><span class="score-pill {pass_fail_class(finance_condition)}">{pass_fail_label(finance_condition)}</span></div>
      </div>
      <div style="font-size:0.78rem; font-weight:700; color:#888; margin-bottom:8px;">점수 평가</div>
      <div class="score-grid">
        <div class="score-item"><span class="s-label">기술력</span><span class="score-pill s-num">{score_technology}/30</span></div>
        <div class="score-item"><span class="s-label">시장성</span><span class="score-pill s-num">{score_market}/25</span></div>
        <div class="score-item"><span class="s-label">경쟁력</span><span class="score-pill s-num">{score_competitiveness}/25</span></div>
        <div class="score-item"><span class="s-label">실적</span><span class="score-pill s-num">{score_traction}/20</span></div>
      </div>
      <div class="verdict-box">
        <div class="v-label">최종 투자 의견</div>
        <div class="v-val {theme['verdict_text_class']}">{escape(verdict)}</div>
      </div>
    </div>
  </div>

  <div class="main-grid">
    <div class="card">
      <div class="card-title"><div class="icon gray">📌</div> 한계점</div>
      <div class="limit-list">
        {render_limit_chips(limit_items)}
      </div>
    </div>

    <div class="card span2">
      <div class="card-title"><div class="icon blue">📚</div> 참고 자료</div>
      <div class="ref-list">
        {render_references(references)}
      </div>
    </div>
  </div>
</section>
""".strip()


def build_multi_company_html(
    sections: list[str], page_title: str = "로보틱스 스타트업 투자 보고서"
) -> str:
    """
    여러 스타트업 섹션을 하나의 HTML 페이지로 합친다.

    - 상단에 공통 헤더 및 목차
    - 본문에 company-section들을 순서대로 렌더링
    - 스타일은 원래 app 버전의 CSS를 그대로 사용
    """
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{escape(page_title)}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
      background: #f0f2f5;
      color: #1a1a2e;
      padding: 32px 24px;
    }}
    .page-header {{
      max-width: 1400px;
      margin: 0 auto 24px auto;
      background: #1a1a2e;
      color: white;
      border-radius: 16px;
      padding: 28px 32px;
    }}
    .page-header h1 {{ font-size: 1.8rem; margin-bottom: 8px; }}
    .page-header p {{ opacity: 0.75; font-size: 0.95rem; }}

    .toc {{
      max-width: 1400px;
      margin: 0 auto 24px auto;
      background: white;
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    .toc h2 {{ font-size: 1rem; margin-bottom: 12px; }}
    .toc ul {{ padding-left: 18px; }}
    .toc li {{ margin-bottom: 6px; }}
    .toc a {{ color: #0f3460; text-decoration: none; font-weight: 600; }}

    .company-section {{
      max-width: 1400px;
      margin: 0 auto 28px auto;
      padding-bottom: 28px;
      border-bottom: 2px dashed #d7dce3;
    }}
    .company-section:last-child {{ border-bottom: none; }}

    .report-header {{
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
      color: white;
      border-radius: 16px;
      padding: 36px 40px;
      margin-bottom: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 20px;
    }}
    .report-header h1 {{ font-size: 1.7rem; font-weight: 700; margin-bottom: 6px; }}
    .report-header p  {{ font-size: 0.9rem; opacity: 0.65; }}

    .header-badge {{
      background: rgba(255,255,255,0.12);
      border: 1px solid rgba(255,255,255,0.2);
      border-radius: 12px;
      padding: 18px 28px;
      text-align: center;
      min-width: 160px;
    }}

    .header-badge .score {{
      font-size: 2.4rem;
      font-weight: 800;
      line-height: 1;
    }}
    .score-blue {{ color: #1976d2; }}
    .score-green {{ color: #2e7d32; }}
    .score-orange {{ color: #ef6c00; }}
    .score-danger {{ color: #c62828; }}

    .header-badge .score-label {{
      font-size: 0.75rem;
      opacity: 0.7;
      margin-top: 4px;
    }}

    .verdict-tag {{
      display: inline-block;
      margin-top: 10px;
      color: white;
      font-size: 0.8rem;
      font-weight: 700;
      padding: 4px 14px;
      border-radius: 20px;
    }}
    .tag-blue {{ background: #1976d2; }}
    .tag-green {{ background: #2e7d32; }}
    .tag-orange {{ background: #ef6c00; }}
    .tag-danger {{ background: #c62828; }}

    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .strip-card {{
      background: white;
      border-radius: 12px;
      padding: 20px 22px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    .strip-card .label {{
      font-size: 0.72rem;
      font-weight: 600;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 8px;
    }}
    .strip-card .value {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #1a1a2e;
    }}

    .tag-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 4px;
    }}
    .tag {{
      font-size: 0.78rem;
      padding: 3px 10px;
      border-radius: 20px;
      font-weight: 500;
    }}
    .tag.green  {{ background: #e8f5e9; color: #2e7d32; }}
    .tag.red    {{ background: #fce4ec; color: #c62828; }}

    .main-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
      margin-bottom: 16px;
    }}

    .card {{
      background: white;
      border-radius: 12px;
      padding: 22px 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    .card.span2 {{ grid-column: span 2; }}

    .card-title {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.78rem;
      font-weight: 700;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #f0f2f5;
    }}
    .card-title .icon {{
      width: 26px;
      height: 26px;
      border-radius: 7px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 0.9rem;
    }}

    .icon.blue   {{ background: #e3f2fd; }}
    .icon.purple {{ background: #f3e5f5; }}
    .icon.green  {{ background: #e8f5e9; }}
    .icon.orange {{ background: #fff3e0; }}
    .icon.red    {{ background: #fce4ec; }}
    .icon.teal   {{ background: #e0f2f1; }}
    .icon.gray   {{ background: #f5f5f5; }}

    .info-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 7px 0;
      border-bottom: 1px solid #f8f8f8;
      font-size: 0.88rem;
      gap: 12px;
    }}
    .info-row:last-child {{ border-bottom: none; }}
    .info-row .key {{ color: #999; font-size: 0.82rem; }}
    .info-row .val {{ font-weight: 600; color: #1a1a2e; text-align: right; }}

    .bullet-list {{ list-style: none; }}
    .bullet-list li {{
      font-size: 0.88rem;
      color: #444;
      padding: 5px 0 5px 14px;
      position: relative;
      line-height: 1.5;
      border-bottom: 1px solid #f8f8f8;
    }}
    .bullet-list li:last-child {{ border-bottom: none; }}
    .bullet-list li::before {{
      content: '▸';
      position: absolute;
      left: 0;
      color: #0f3460;
      font-size: 0.75rem;
      top: 6px;
    }}

    .tech-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .tech-item {{
      background: #f8f9fa;
      border-radius: 8px;
      padding: 10px 14px;
    }}
    .tech-item .t-label {{
      font-size: 0.72rem;
      color: #999;
      margin-bottom: 4px;
      font-weight: 600;
    }}
    .tech-item .t-val {{
      font-size: 0.85rem;
      font-weight: 600;
      color: #1a1a2e;
    }}

    .comp-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    .comp-table th {{
      background: #f0f2f5;
      padding: 9px 12px;
      text-align: left;
      font-weight: 700;
      color: #555;
      font-size: 0.78rem;
    }}
    .comp-table td {{
      padding: 9px 12px;
      border-bottom: 1px solid #f5f5f5;
      color: #444;
    }}
    .comp-table tr:last-child td {{ border-bottom: none; }}
    .comp-table td:first-child {{
      font-weight: 600;
      color: #1a1a2e;
    }}
    .highlight-col {{ background: #fff9c4 !important; }}

    .score-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }}
    .score-item {{
      background: #f8f9fa;
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }}
    .score-item .s-label {{ font-size: 0.82rem; color: #555; }}
    .score-pill {{
      font-size: 0.82rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      white-space: nowrap;
    }}
    .s-pass {{ background: #e8f5e9; color: #2e7d32; }}
    .s-fail {{ background: #fce4ec; color: #c62828; }}
    .s-num  {{ background: #e3f2fd; color: #1565c0; }}

    .verdict-box {{
      background: linear-gradient(135deg, #0f3460, #16213e);
      color: white;
      border-radius: 10px;
      padding: 18px 22px;
      text-align: center;
      margin-top: 14px;
    }}
    .verdict-box .v-label {{
      font-size: 0.75rem;
      opacity: 0.65;
      margin-bottom: 6px;
    }}
    .verdict-box .v-val {{
      font-size: 1.4rem;
      font-weight: 800;
    }}

    .verdict-blue {{ color: #42a5f5; }}
    .verdict-green {{ color: #4caf50; }}
    .verdict-orange {{ color: #ff9800; }}
    .verdict-danger {{ color: #f44336; }}

    .risk-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .risk-item {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      background: #fff8f8;
      border-left: 3px solid #e94560;
      border-radius: 0 8px 8px 0;
      padding: 10px 12px;
      font-size: 0.87rem;
    }}
    .risk-item .r-icon {{
      font-size: 1rem;
      flex-shrink: 0;
    }}
    .risk-item .r-text {{
      color: #444;
      line-height: 1.4;
    }}
    .risk-item .r-title {{
      font-weight: 700;
      color: #c62828;
      margin-bottom: 2px;
    }}

    .ref-list {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .ref-item {{
      background: #f8f9fa;
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 0.83rem;
      color: #555;
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      word-break: break-word;
    }}
    .ref-item span {{ opacity: 0.5; }}
    .ref-item a {{
      color: #555;
      text-decoration: none;
      word-break: break-all;
    }}

    .limit-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .limit-chip {{
      background: #fff3e0;
      border: 1px solid #ffe082;
      color: #e65100;
      font-size: 0.82rem;
      font-weight: 500;
      padding: 5px 14px;
      border-radius: 20px;
    }}

    .market-stats {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }}
    .m-stat {{
      background: #f8f9fa;
      border-radius: 8px;
      padding: 12px 14px;
      text-align: center;
    }}
    .m-stat .m-num {{
      font-size: 1rem;
      font-weight: 800;
      color: #0f3460;
      line-height: 1.3;
    }}
    .m-stat .m-unit {{
      font-size: 0.72rem;
      color: #999;
      margin-top: 4px;
    }}

    @media (max-width: 900px) {{
      .main-grid {{ grid-template-columns: 1fr 1fr; }}
      .summary-strip {{ grid-template-columns: 1fr 1fr; }}
      .ref-list {{ grid-template-columns: 1fr 1fr; }}
    }}

    @media (max-width: 600px) {{
      .main-grid {{ grid-template-columns: 1fr; }}
      .summary-strip {{ grid-template-columns: 1fr; }}
      .report-header {{ flex-direction: column; }}
      .score-grid {{ grid-template-columns: 1fr; }}
      .market-stats {{ grid-template-columns: 1fr 1fr; }}
      .ref-list {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page-header">
    <h1>{escape(page_title)}</h1>
    <p>여러 스타트업의 투자 분석 결과를 하나의 HTML 페이지에서 스크롤로 확인할 수 있습니다.</p>
  </div>

  <div class="toc">
    <h2>목차</h2>
    <ul>
      {''.join(f'<li><a href="#company-{i+1}">{i+1}번 스타트업 보고서</a></li>' for i in range(len(sections)))}
    </ul>
  </div>

  {"".join(sections)}
</body>
</html>
""".strip()


def final_report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    section_html = _build_company_report_section(state)
    full_html = build_multi_company_html([section_html], page_title="로보틱스 스타트업 투자 보고서")
    state["final_report"] = FinalReport(html=full_html)
    return state


class FinalReportAgent:
    def run(self, state: Dict[str, Any]) -> FinalReport:
        out = final_report_node(state)
        report = out.get("final_report")
        if isinstance(report, FinalReport):
            return report
        if hasattr(report, "model_dump"):
            return FinalReport.model_validate(report)
        raise ValueError("final_report was not produced.")

    def render_pdf(self, report: FinalReport, output_path: str) -> None:
        """
        FinalReport(HTML 기반)를 지정된 경로의 PDF 파일로 저장한다.

        사용 예:
            agent = FinalReportAgent()
            report = agent.run(state)
            agent.render_pdf(report, \"final_report.pdf\")

        내부적으로 utils.html_to_pdf()를 사용하며, weasyprint 미설치 시 RuntimeError를 발생시킨다.
        """
        html_to_pdf(report.html, output_path)

