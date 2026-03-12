from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from agents.startup_search.agent import StartupSearchAgent

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

    # 로컬 import 해결을 위해 sys.path 삽입
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
    startup_search 결과가 need_query_refinement=True일 때만 실행되는 노드.
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
            "retrieval_feedback": {
                "result_count": len(retrieved_docs),
                "top_scores": top_scores,
                "noisy_results": False,
                "failure_reason": "startup_search need_query_refinement=True",
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
    selected = set(state.get("selected_startup_names") or [])

    # technical-summary 폴더 경로(하이픈) 사용
    agent_dir = _project_root() / "agents" / "technical-summary"

    outs: List[Dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue

        # 기술 요약 입력: startup_search의 technology 섹션을 그대로 전달
        payload = {"startup_name": name, "technology_info": (p.get("technology") or {})}
        out = _call_local_agent_module(agent_dir, "agent.py", "TechSummaryAgent", payload)
        outs.append(out)

    return {"technical_summaries": outs}


def _should_web_search_for_profile(profile: Dict[str, Any], cfg: SupervisorConfig) -> bool:
    # fields가 너무 빈약하면 web search로 보강
    return _count_nonempty_profile_fields(profile) < cfg.web_search_min_profile_fields


def node_web_search(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    if not cfg.enable_web_search:
        return {"web_search_results": []}

    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    selected = set(state.get("selected_startup_names") or [])

    agent_dir = _project_root() / "agents" / "web-search"

    outs: List[Dict[str, Any]] = []
    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue
        if not _should_web_search_for_profile(p, cfg):
            continue
        out = _call_local_agent_module(agent_dir, "agent.py", "WebSearchAgent", {"startup_name": name})
        outs.append(out)

    return {"web_search_results": outs}


def node_finalize_report(state: SupervisorState, cfg: SupervisorConfig) -> SupervisorState:
    """
    LLM 없이도 후속 에이전트가 소비할 수 있도록, state 내용을 '근거 중심'으로 묶어서 출력.
    (요약/해석 최소화)
    """
    ss = state.get("startup_search") or {}
    profiles = ss.get("startup_profiles") or []
    selected = set(state.get("selected_startup_names") or [])

    tech_summaries = state.get("technical_summaries") or []
    web_results = state.get("web_search_results") or []

    # 빠른 lookup
    tech_by_name = {o.get("startup_name"): o for o in tech_summaries if isinstance(o, dict)}
    web_by_name = {o.get("startup_name"): o for o in web_results if isinstance(o, dict)}

    lines: List[str] = []
    lines.append("## Supervisor Output")
    lines.append(f"- user_query: {state.get('user_query','')}")
    if state.get("refined_query"):
        lines.append(f"- refined_query: {state.get('refined_query')}")
    lines.append("")
    lines.append("## Startup candidates")

    for p in profiles:
        if not isinstance(p, dict):
            continue
        name = (p.get("company_name") or "").strip()
        if name not in selected:
            continue

        lines.append(f"### {name}")
        lines.append(f"- score: {p.get('score')}")
        lines.append(f"- pages: {p.get('pages')}")
        ov = p.get("company_overview") or {}
        tech = p.get("technology") or {}
        perf = p.get("performance") or {}

        # 원문 기반 필드(가능한 그대로)
        if ov:
            lines.append(f"- company_overview: {ov}")
        if tech:
            lines.append(f"- technology: {tech}")
        if perf:
            lines.append(f"- performance: {perf}")

        refs = p.get("references") or []
        if refs:
            lines.append("- references:")
            for r in refs:
                lines.append(f"  - {r}")

        ts = tech_by_name.get(name)
        if ts:
            lines.append("- technical_summary:")
            # messages 같은 부가 필드는 그대로 둬도 되지만, 보고서에는 핵심만
            for k in (
                "core_technology",
                "tech_summary",
                "tech_strengths",
                "tech_weaknesses",
                "tech_differentiation",
                "patent_count",
                "rd_team_size",
            ):
                if k in ts:
                    lines.append(f"  - {k}: {ts.get(k)}")

        ws = web_by_name.get(name)
        if ws:
            lines.append("- web_search:")
            for k in ("market", "technology", "competition", "performance"):
                if k in ws:
                    lines.append(f"  - {k}: {ws.get(k)}")

        lines.append("")

    return {"final_report": "\n".join(lines).strip()}


# ============================================================
# Graph builder
# ============================================================


def build_supervisor_graph(cfg: Optional[SupervisorConfig] = None):
    cfg = cfg or SupervisorConfig()

    g = StateGraph(SupervisorState)

    g.add_node("startup_search", lambda s: node_startup_search(s, cfg))
    g.add_node("query_refinement", lambda s: node_query_refinement(s, cfg))
    g.add_node("select_candidates", lambda s: node_select_candidates(s, cfg))
    g.add_node("technical_summary", lambda s: node_technical_summary(s, cfg))
    g.add_node("web_search", lambda s: node_web_search(s, cfg))
    g.add_node("finalize", lambda s: node_finalize_report(s, cfg))

    g.set_entry_point("startup_search")

    # startup_search -> (need refine?) -> query_refinement -> startup_search
    def route_after_startup_search(state: SupervisorState) -> str:
        ss = state.get("startup_search") or {}
        if ss.get("need_query_refinement") and not state.get("refined_query"):
            return "query_refinement"
        return "select_candidates"

    g.add_conditional_edges(
        "startup_search",
        route_after_startup_search,
        {
            "query_refinement": "query_refinement",
            "select_candidates": "select_candidates",
        },
    )

    # refinement 후에는 refined_query로 다시 검색
    g.add_edge("query_refinement", "startup_search")

    # downstream
    g.add_edge("select_candidates", "technical_summary")
    g.add_edge("technical_summary", "web_search")
    g.add_edge("web_search", "finalize")
    g.add_edge("finalize", END)

    return g.compile()

