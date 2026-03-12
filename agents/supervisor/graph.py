from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from agents.startup_search.agent import StartupSearchAgent
from agents.investment_decision_agent import InvestmentDecisionAgent
from agents.market_evaluation_agent import MarketEvaluationAgent
from agents.final_report_agent.agent import _build_company_report_section, build_multi_company_html

from .state import SupervisorState


def _project_root() -> Path:
    # agents/supervisor/graph.py -> agents/supervisor -> agents -> project root
    return Path(__file__).resolve().parents[2]


@dataclass
class SupervisorConfig:
    """
    Supervisor 실행 설정 (LangGraph node 실행에 필요한 최소 파라미터).
    """

    # startup_search
    startup_search_k: int = 10

    # 후보 선정
    max_startups_for_downstream: int = 3

    # web search 조건
    enable_web_search: bool = True
    web_search_min_profile_fields: int = 2  # profile에서 의미 있게 채워진 필드가 너무 적으면 web search

    # technical summary 조건
    enable_technical_summary: bool = True


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _count_nonempty_profile_fields(profile: Dict[str, Any]) -> int:
    """
    web search 트리거용: startup_search profile에서 채워진 필드 수를 대충 계산.
    (요약/해석 금지. 단순 '값이 비었는지'만 본다.)
    """
    cnt = 0

    overview = profile.get("company_overview") or {}
    tech = profile.get("technology") or {}
    perf = profile.get("performance") or {}
    diff = profile.get("differentiation") or {}
    sal = profile.get("strengths_and_limitations") or {}

    scalar_fields = [
        overview.get("industry"),
        overview.get("main_product"),
        overview.get("investment_status"),
        overview.get("website"),
        tech.get("maturity"),
        tech.get("patent_ip"),
        diff.get("description"),
        perf.get("customers_references"),
        perf.get("sales_growth"),
        perf.get("adoption_cases"),
    ]
    for v in scalar_fields:
        if isinstance(v, str) and v.strip() and v.strip() not in {"정보 없음"}:
            cnt += 1

    # list fields
    for lst in (sal.get("strengths"), sal.get("limitations"), diff.get("core_advantages"), profile.get("references")):
        if isinstance(lst, list) and any(isinstance(x, str) and x.strip() for x in lst):
            cnt += 1

    return cnt


def _select_startups(state: SupervisorState, cfg: SupervisorConfig) -> List[str]:
    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    names: List[str] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        n = (p.get("company_name") or "").strip()
        if not n:
            continue
        if n.lower() == "unknown":
            continue
        if n not in names:
            names.append(n)
        if len(names) >= cfg.max_startups_for_downstream:
            break
    return names


def _load_module_from_file(module_name: str, file_path: Path):
    """
    특정 파일을 '유니크 모듈명'으로 로드.
    technical-summary / web-search 처럼 하이픈 폴더를 직접 import 못 하는 문제를 우회.
    """
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _call_local_agent_module(agent_dir: Path, agent_py: str, class_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    하이픈 폴더(technical-summary, web-search)의 agent.py를 직접 로드하여 실행.

    주의:
    - 해당 폴더 내부는 `from state import ...` 같은 로컬 import를 사용하므로,
      호출 중에만 agent_dir을 sys.path에 넣어서 로컬 import가 풀리게 한다.
    - state.py / prompts.py 등 공통 이름 충돌을 줄이기 위해, 호출 후 sys.path는 원복한다.
    """
    agent_dir = agent_dir.resolve()
    agent_file = agent_dir / agent_py
    if not agent_file.exists():
        raise FileNotFoundError(str(agent_file))

    for mod_name in ["state", "prompts"]:
        if mod_name in sys.modules:
            sys.modules.pop(mod_name, None)

    sys.path.insert(0, str(agent_dir))
    try:
        mod = _load_module_from_file(f"_dynamic_{agent_dir.name}_{class_name}", agent_file)
        cls = getattr(mod, class_name)
        inst = cls()
        out = inst(payload)
        if not isinstance(out, dict):
            return {"error": f"{class_name} output was not a dict", "raw_output": out}
        return out
    finally:
        # sys.path 원복
        if sys.path and sys.path[0] == str(agent_dir):
            sys.path.pop(0)


# ============================================================
# LangGraph nodes
# ============================================================


def node_startup_search(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    q = (state.get("refined_query") or state.get("user_query") or "").strip()
    if not q:
        return {"error": "user_query is empty"}

    agent = StartupSearchAgent()
    out = agent.run(q, k=cfg.startup_search_k)
    return {"startup_search": out.model_dump()}


def node_query_refinement(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    """
    질문 정제 에이전트 노드.
    - 최초 진입 시에는 retrieval_feedback 없이 raw_question만 사용
    - 필요하다면 startup_search 결과를 기반으로 feedback을 줄 수 있음
    """
    raw_question = (state.get("user_query") or "").strip()
    if not raw_question:
        return {"error": "user_query is empty"}

    ss = state.get("startup_search") or {}
    retrieved_docs = ss.get("retrieved_docs") or []
    top_scores = []
    for d in retrieved_docs[:3]:
        if isinstance(d, dict) and isinstance(d.get("score"), (int, float)):
            top_scores.append(float(d["score"]))

    # agents.query_refinement는 정상 import 가능(상대 import로 수정됨)
    from agents.query_refinement.agent import QueryRefinementAgent

    agent = QueryRefinementAgent()
    result = agent.refine(
        {
            "raw_question": raw_question,
            # 최초 실행 시에는 retrieved_docs가 비어있어도 허용
            "retrieval_feedback": {
                "result_count": len(retrieved_docs),
                "top_scores": top_scores,
                "noisy_results": False,
                "failure_reason": (
                    "initial_query_refinement"
                    if not retrieved_docs
                    else "startup_search feedback for refinement"
                ),
            },
        }
    )
    return {
        "refined_query": result.get("refined_query", raw_question),
        "reformulated_queries": result.get("reformulated_queries", []),
        "retry_strategy": result.get("retry_strategy", ""),
    }


def node_select_candidates(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    names = _select_startups(state, cfg)
    return {"selected_startup_names": names}


def node_technical_summary(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    if not cfg.enable_technical_summary:
        return {"technical_summaries": []}

    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    # 후보선택 노드를 생략할 수 있으므로, 여기서 상위 N개를 직접 선택
    selected = set(_select_startups(state, cfg))

    # technical-summary 폴더 경로(하이픈) 사용
    agent_dir = _project_root() / "agents" / "technical-summary"

    outs: List[Dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue

        # startup_search에서 추출한 구조화 필드 + raw_text를 함께 전달
        overview = p.get("company_overview") or {}
        tech_section = p.get("technology") or {}
        sal = p.get("strengths_and_limitations") or {}
        diff = p.get("differentiation") or {}
        perf = p.get("performance") or {}
        raw_text = p.get("merged_raw_text") or ""

        technology_info = {
            "structured_fields": {
                "company_overview": overview,
                "technology": tech_section,
                "strengths_and_limitations": sal,
                "differentiation": diff,
                "performance": perf,
            },
            "raw_text": raw_text,
        }

        payload = {"startup_name": name, "technology_info": technology_info}
        raw = _call_local_agent_module(agent_dir, "agent.py", "TechSummaryAgent", payload)

        # TechSummaryAgent는 tech_strengths / tech_weaknesses / tech_differentiation 키를 사용한다.
        # downstream(investment_decision, final_report)에서는 strengths / limitations / differentiation을 기대하므로
        # 여기서 스키마를 한 번 정규화해 둔다.
        if isinstance(raw, dict):
            norm: Dict[str, Any] = {
                "startup_name": raw.get("startup_name", name),
                # LLM 출력 그대로 유지
                "core_technology": raw.get("core_technology"),
                "tech_summary": raw.get("tech_summary"),
                # downstream 호환 필드
                "strengths": raw.get("tech_strengths") or raw.get("strengths") or [],
                "limitations": raw.get("tech_weaknesses") or raw.get("limitations") or [],
                "differentiation": raw.get("tech_differentiation") or raw.get("differentiation"),
                # 참고 자료 등 기타 필드는 그대로 전달
                "references": raw.get("references", []),
            }
            outs.append(norm)
        else:
            outs.append({"startup_name": name})

    return {"technical_summaries": outs}


def _should_web_search_for_profile(profile: Dict[str, Any], cfg: SupervisorConfig) -> bool:
    # fields가 너무 빈약하면 web search로 보강
    return _count_nonempty_profile_fields(profile) < cfg.web_search_min_profile_fields


def node_web_search(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    if not cfg.enable_web_search:
        return {"web_search_results": []}

    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    # 후보선택 노드를 생략할 수 있으므로, 여기서 상위 N개를 직접 선택
    selected = set(_select_startups(state, cfg))

    agent_dir = _project_root() / "agents" / "web-search"

    outs: List[Dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue

        # 과거에는 profile 필드가 너무 비어 있을 때만 web-search를 돌렸지만,
        # 투자 점수/보고서에서 최신 정보를 적극 활용하기 위해
        # 선택된 후보들에 대해서는 항상 WebSearchAgent를 호출한다.
        out = _call_local_agent_module(agent_dir, "agent.py", "WebSearchAgent", {"startup_name": name})
        outs.append(out)

    return {"web_search_results": outs}


def node_market_evaluation(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    """
    시장성 평가 에이전트 노드.

    - startup_search + 기술 요약을 바탕으로, market_eval vector store에서
      시장/산업 보고서를 조회하고 LLM으로 시장성을 정리한다.
    """
    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    selected = set(_select_startups(state, cfg))

    if not profiles or not selected:
        return {"market_evaluations": []}

    outs: List[Dict[str, Any]] = []
    agent = MarketEvaluationAgent()

    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue

        ov = p.get("company_overview") or {}
        tech = p.get("technology") or {}

        # 시장성 평가 쿼리: 사용자의 원 질문 + 회사 개요/기술 요약 요약본을 합쳐서 전달
        user_query = (state.get("user_query") or "").strip()
        market_query = user_query or (ov.get("industry") or "") or "로봇 시장성 평가"

        company_context_parts = []
        if ov.get("industry"):
            company_context_parts.append(str(ov.get("industry")))
        if ov.get("main_product"):
            company_context_parts.append(str(ov.get("main_product")))
        if tech.get("summary"):
            company_context_parts.append(str(tech.get("summary")))
        company_context = " / ".join(part.strip() for part in company_context_parts if part)

        try:
            out = agent(
                {
                    "startup_name": name,
                    "market_query": market_query,
                    "company_context": company_context,
                    "top_k": 4,
                }
            )
            outs.append(out)
        except Exception as exc:
            outs.append(
                {
                    "startup_name": name,
                    "market_query": market_query,
                    "search_query": "",
                    "market_summary": f"시장성 평가 실패: {exc}",
                    "market_size": "자료 부족",
                    "growth_drivers": [],
                    "competition_analysis": "자료 부족",
                    "customer_adoption": "자료 부족",
                    "key_risks": ["시장성 평가 에이전트 오류"],
                    "evidence": [],
                }
            )

    return {"market_evaluations": outs}

def node_finalize_report(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    """
    최종 HTML 보고서 생성.

    - startup_search + 기술 요약 + (간단한) 웹 검색 정보를 바탕으로
      investment_decision_agent / final_report_agent를 사용해 예쁜 HTML을 만든다.
    - 현재는 상위 1개 스타트업 기준 단일 보고서를 생성한다.
    """
    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    selected_names = _select_startups(state, cfg)
    if not profiles or not selected_names:
        return {
            "final_report": f"no startup candidates for query: {state.get('user_query','')}",
        }

    # lookup 테이블
    tech_summaries = state.get("technical_summaries") or []
    web_results = state.get("web_search_results") or []
    market_evals = state.get("market_evaluations") or []
    tech_by_name = {o.get("startup_name"): o for o in tech_summaries if isinstance(o, dict)}
    web_by_name = {o.get("startup_name"): o for o in web_results if isinstance(o, dict)}
    market_by_name = {o.get("startup_name"): o for o in market_evals if isinstance(o, dict)}

    def _as_str(v: Any) -> str:
        return str(v) if v is not None else ""

    # 각 회사별로 (투자 점수, inv_state)를 모아서,
    # 최종 HTML 섹션은 투자 점수 내림차순으로 정렬해 렌더링한다.
    scored_states: List[Dict[str, Any]] = []

    for primary_name in selected_names[: cfg.max_startups_for_downstream]:
        # 대상 회사 프로필 찾기
        profile: Dict[str, Any] | None = None
        for p in profiles:
            if isinstance(p, dict) and (p.get("company_name") or "").strip() == primary_name:
                profile = p
                break
        if profile is None:
            continue

        ov = profile.get("company_overview") or {}
        tech_section = profile.get("technology") or {}

        # one_liner: 산업/주생산품 또는 기술 요약 첫 줄
        one_liner_parts = []
        if ov.get("industry"):
            one_liner_parts.append(_as_str(ov.get("industry")))
        if ov.get("main_product"):
            one_liner_parts.append(_as_str(ov.get("main_product")))
        if one_liner_parts:
            one_liner = " / ".join(one_liner_parts)
        else:
            summary_text = _as_str(tech_section.get("summary", ""))
            one_liner = summary_text.split("\n")[0][:120] if summary_text else ""

        # 웹 검색 결과에서 market / competition / performance 정보 단순 결합
        ws = web_by_name.get(primary_name) or {}
        market_facts = ws.get("market") or []
        competition_facts = ws.get("competition") or []
        performance_facts = ws.get("performance") or []

        # 시장성 평가는 market_evaluation_agent 결과를 우선 사용하고,
        # 부족한 경우에만 web-search facts로 보완한다.
        me = market_by_name.get(primary_name) or {}

        # startup_search에서 온 RAG 참고 URL들
        rag_refs = []
        for url in profile.get("references") or []:
            if isinstance(url, str) and url.strip():
                rag_refs.append(
                    {
                        "title": url.strip(),
                        "url": url.strip(),
                        "source_type": "rag",
                    }
                )

        exploration_summary = {
            "competitor_summary": me.get("competition_analysis")
            or ("\n".join(competition_facts) if competition_facts else ""),
            "traction_summary": me.get("customer_adoption")
            or ("\n".join(performance_facts) if performance_facts else ""),
            # RAG 및 웹/시장 보고서에서 온 참고 자료를 함께 보관 (FinalReport에서 dedup)
            "references": rag_refs,
        }

        market_assessment = {
            "market_size": me.get("market_size") or (market_facts[0] if market_facts else "정보 부족"),
            "growth_drivers": me.get("growth_drivers")
            or (market_facts[1:3] if len(market_facts) > 1 else []),
            "target_industries": me.get("target_industries") or [],
            "scalability": "정보 부족",
            "commercialization_risk": (", ".join(me.get("key_risks", [])) or None),
            "references": rag_refs,
        }

        # web-search 기반 최신 고객/투자 이력도 startup_profile에 반영
        customers: list[str] = []
        fundraising: list[str] = []
        for item in performance_facts:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            if any(kw in text for kw in ["투자", "유치", "시리즈", "프리 IPO", "IPO", "라운드"]):
                fundraising.append(text)
            if any(kw in text for kw in ["고객", "레퍼런스", "도입", "파일럿", "파트너", "협력"]):
                customers.append(text)

        def _dedup_str_list(values: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out

        startup_profile = {
            "company_name": primary_name,
            "domain": "Robotics",
            "one_liner": one_liner,
            "headquarters": ov.get("region"),
            "founded_year": None,
            "representative": ov.get("representative"),
            "products": [ov.get("main_product")] if ov.get("main_product") else [],
            "customers": _dedup_str_list(customers),
            "fundraising_history": _dedup_str_list(fundraising),
            "patents": [],
        }
        fy = ov.get("founded_year")
        if isinstance(fy, str):
            try:
                startup_profile["founded_year"] = int(fy)
            except ValueError:
                startup_profile["founded_year"] = None

        # 기술 요약: TechSummaryAgent 결과 사용
        tech_summary = tech_by_name.get(primary_name) or {}

        inv_state: Dict[str, Any] = {
            "user_query": state.get("user_query", ""),
            "startup_profile": startup_profile,
            "exploration_summary": exploration_summary,
            "technology_summary": tech_summary,
            "market_assessment": market_assessment,
        }

        try:
            inv_agent = InvestmentDecisionAgent()
            decision = inv_agent.run(inv_state)
            if hasattr(decision, "model_dump"):
                inv_state["investment_decision"] = decision.model_dump()
            else:
                inv_state["investment_decision"] = decision

            # 투자 점수(총점)를 꺼내서 정렬용 스코어로 사용
            total_score = 0.0
            try:
                weighted = getattr(decision, "weighted_score", None)
                if weighted is not None and hasattr(weighted, "total"):
                    total_score = float(weighted.total)
                else:
                    raw_dec = inv_state.get("investment_decision") or {}
                    w = raw_dec.get("weighted_score") or {}
                    total_score = float(w.get("total", 0.0))
            except Exception:
                total_score = 0.0

            scored_states.append({"score": total_score, "state": inv_state})
        except Exception as exc:
            # 개별 회사 실패 시에도 나머지 회사들은 계속 진행
            scored_states.append(
                {
                    "score": 0.0,
                    "state": {
                        "user_query": state.get("user_query", ""),
                        "startup_profile": {"company_name": primary_name},
                        "exploration_summary": {},
                        "technology_summary": {},
                        "market_assessment": {},
                        "investment_decision": {
                            "weighted_score": {"total": 0},
                            "verdict": "투자 위험",
                        },
                        "error": f"failed to build state for {primary_name}: {exc}",
                    },
                }
            )

    if not scored_states:
        return {"final_report": f"no startup profiles could be rendered for query: {state.get('user_query','')}"}

    # 투자 점수 내림차순으로 정렬 후, 섹션 인덱스를 1부터 다시 매겨서 렌더링
    scored_states.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    sections: List[str] = []
    for idx, item in enumerate(scored_states, start=1):
        inv_state = item["state"]
        section_html = _build_company_report_section(inv_state, section_index=idx)
        sections.append(section_html)

    full_html = build_multi_company_html(sections, page_title="로보틱스 스타트업 투자 보고서")
    return {"final_report": full_html}


# ============================================================
# Graph builder
# ============================================================


def build_supervisor_graph(cfg: Optional[SupervisorConfig] = None):
    cfg = cfg or SupervisorConfig()

    g = StateGraph(SupervisorState)

    g.add_node("query_refinement", lambda s: node_query_refinement(s, cfg))
    g.add_node("startup_search", lambda s: node_startup_search(s, cfg))
    g.add_node("technical_summary", lambda s: node_technical_summary(s, cfg))
    g.add_node("web_search", lambda s: node_web_search(s, cfg))
    g.add_node("market_evaluation", lambda s: node_market_evaluation(s, cfg))
    g.add_node("finalize", lambda s: node_finalize_report(s, cfg))

    # 1. 항상 질문 정제부터 시작
    g.set_entry_point("query_refinement")

    # 2. 정제된 질의로 startup_search 실행
    g.add_edge("query_refinement", "startup_search")

    # 3. startup_search -> (score 기반으로 계속 진행 여부 결정)
    def route_after_startup_search(state: SupervisorState) -> str:
        ss = state.get("startup_search") or {}
        conf = float(ss.get("search_confidence") or 0.0)
        # 정제된 질의로 검색을 이미 한 번 수행했으므로,
        # score < 0.6이면 여기서 종료, 아니면 후보 선택으로 진행
        if conf < 0.6:
            return "end"
        # conf >= 0.6이면 downstream으로 진행 (후보선택 노드 생략)
        return "technical_summary"

    g.add_conditional_edges(
        "startup_search",
        route_after_startup_search,
        {
            "end": END,
            "technical_summary": "technical_summary",
        },
    )

    # 4. downstream
    g.add_edge("technical_summary", "web_search")
    g.add_edge("web_search", "market_evaluation")
    g.add_edge("market_evaluation", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

