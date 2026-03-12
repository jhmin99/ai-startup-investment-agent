from __future__ import annotations

from collections import defaultdict
import re
from typing import Dict, List, Optional, Tuple

from .retriever import StartupRetriever
from .parser import parse_profile_sections
from .schemas import RetrievedDocument, StartupProfile, StartupSearchOutput
from .utils import (
    calculate_search_confidence,
    extract_company_name_from_text,
    normalize_query,
    should_refine_query,
)


class StartupSearchAgent:
    """
    startup_search 에이전트의 메인 진입점.

    - supervisor/LangGraph 없이 단독 실행 가능
    - run()은 사이드이펙트를 최소화하고 결과를 명확히 반환
    """

    def __init__(self, retriever: Optional[StartupRetriever] = None):
        self.retriever = retriever or StartupRetriever()

    def run(self, user_query: str, k: int = 5) -> StartupSearchOutput:
        normalized = normalize_query(user_query)
        retrieved_docs = self.retriever.search(normalized, k=k)
        profiles = self._build_startup_profiles(retrieved_docs)

        confidence = calculate_search_confidence(retrieved_docs, candidate_count=len(profiles))
        need_refine = should_refine_query(confidence, candidate_count=len(profiles))

        return StartupSearchOutput(
            user_query=user_query,
            normalized_query=normalized,
            retrieved_docs=retrieved_docs,
            startup_profiles=profiles,
            search_confidence=confidence,
            need_query_refinement=need_refine,
        )

    def _build_startup_profiles(self, retrieved_docs: List[RetrievedDocument]) -> List[StartupProfile]:
        """
        검색 결과 chunk들을 회사별로 묶어서 StartupProfile 리스트를 구성.

        - 같은 회사 chunk가 여러 개면 1개의 profile로 merge
        - 파싱이 실패해도 raw_texts/merged_raw_text/metadata_list는 항상 유지
        """
        # 1) 우선 회사명(추정 포함)으로 그룹핑
        resolved = [self._resolve_company_name(doc, retrieved_docs) for doc in retrieved_docs]

        groups: Dict[str, List[Tuple[RetrievedDocument, dict]]] = defaultdict(list)
        for doc, md, company_name in resolved:
            groups[str(company_name)].append((doc, md))

        profiles: List[StartupProfile] = []
        for company_name, items in groups.items():
            # page/chunk_index 기준으로 정렬 후 merge (토트처럼 chunk 순서 섞임 방지)
            items_sorted = self._sort_doc_items(items)
            docs = [d for d, _ in items_sorted]
            md_list = [m for _, m in items_sorted]
            raw_texts = [(d.content or "").strip() for d in docs if (d.content or "").strip()]
            merged_raw_text = "\n\n".join(raw_texts).strip()

            # pages / source_file
            pages = []
            source_file = None
            for m in md_list:
                if source_file is None and m.get("source_file"):
                    source_file = m.get("source_file")
                p = m.get("page")
                if isinstance(p, int):
                    pages.append(p)
            pages = sorted(set(pages))

            score = max((d.score for d in docs), default=0.0)

            # 1차: merged_raw_text 기준 섹션 파싱
            overview, technology, strengths_limits, diff, perf, refs_from_text = parse_profile_sections(merged_raw_text)

            # 2차: ingestion 단계에서 metadata.references에 넣어둔 PDF 단위 참고 URL 병합
            refs_from_meta: List[str] = []
            for m in md_list:
                for u in (m.get("references") or []):
                    if isinstance(u, str) and u.strip():
                        refs_from_meta.append(u.strip())

            # 중복 제거(순서 유지)
            all_refs: List[str] = []
            seen_refs: set[str] = set()
            for u in list(refs_from_text or []) + refs_from_meta:
                if u not in seen_refs:
                    seen_refs.add(u)
                    all_refs.append(u)

            # 디버깅: 참고 URL 파싱 결과를 로그로 남긴다.
            try:
                if all_refs:
                    print("\n[startup_search] references parsed for company:", company_name)
                    for r in all_refs:
                        print("  -", r)
                else:
                    # 참고 블록이 안 잡히는 경우 raw 텍스트 끝부분을 확인할 수 있도록 짧게 출력
                    tail = merged_raw_text[-500:].replace("\n", "\\n")
                    print(f"\n[startup_search] no references for company: {company_name}")
                    print("[startup_search] merged_raw_text tail:", tail)
            except Exception:
                # 로깅 실패 시 검색 로직에는 영향 주지 않음
                pass

            profiles.append(
                StartupProfile(
                    company_name=company_name,
                    score=score,
                    source_file=source_file,
                    pages=pages,
                    company_overview=overview,
                    technology=technology,
                    strengths_and_limitations=strengths_limits,
                    differentiation=diff,
                    performance=perf,
                    references=all_refs,
                    raw_texts=raw_texts,
                    merged_raw_text=merged_raw_text,
                    metadata_list=md_list,
                )
            )

        # 2) unknown 그룹이 남아있으면 가능한 경우 기존 회사로 병합
        profiles = self._merge_unknown_profiles(profiles)

        # 점수 내림차순
        profiles.sort(key=lambda p: p.score, reverse=True)
        return profiles

    def _sort_doc_items(self, items: List[Tuple[RetrievedDocument, dict]]) -> List[Tuple[RetrievedDocument, dict]]:
        """metadata.page, metadata.chunk_index 오름차순 정렬 (없으면 뒤로)."""
        def key(it):
            md = it[1] or {}
            p = md.get("page")
            ci = md.get("chunk_index")
            p = p if isinstance(p, int) else 10**9
            ci = ci if isinstance(ci, int) else 10**9
            return (p, ci)

        return sorted(items, key=key)

    def _resolve_company_name(
        self, doc: RetrievedDocument, all_docs: List[RetrievedDocument]
    ) -> Tuple[RetrievedDocument, dict, str]:
        """
        company_name이 없는 chunk를 무조건 unknown으로 보내지 않고, 주변 문맥으로 추정.

        추정 우선순위:
        1) metadata.company_name
        2) chunk 본문에서 회사명 패턴 추출
        3) (unknown일 때) page / chunk_index 인접 + 섹션 연속성으로 이전 회사에 붙이기
        """
        md = doc.metadata or {}
        name = md.get("company_name") or extract_company_name_from_text(doc.content)
        if name:
            return doc, md, str(name)

        # 주변 문서 기반 추정
        page = md.get("page")
        chunk_index = md.get("chunk_index")
        text = (doc.content or "").strip()

        def looks_like_continuation(t: str) -> bool:
            return bool(
                t.startswith("3.")
                or t.startswith("4.")
                or t.startswith("5.")
                or re.match(r"^\s*[345]\.\s", t)
                or re.search(r"\b(기술적\s*강점\s*및\s*한계|경쟁사\s*대비\s*차별성|실적)\b", t)
            )

        # all_docs를 page/chunk_index 기준으로 정렬해서 이전 known을 찾는다
        sortable: List[Tuple[int, int, RetrievedDocument]] = []
        for d in all_docs:
            m = d.metadata or {}
            p = m.get("page")
            ci = m.get("chunk_index")
            if isinstance(p, int) and isinstance(ci, int):
                sortable.append((p, ci, d))
        sortable.sort()

        if isinstance(page, int) and isinstance(chunk_index, int) and sortable:
            # 현 위치 인덱스 찾기
            cur_pos = None
            for idx, (p, ci, d) in enumerate(sortable):
                if p == page and ci == chunk_index:
                    cur_pos = idx
                    break
            if cur_pos is not None and looks_like_continuation(text):
                # 바로 이전/근처(최대 5개)에서 known company 찾기
                for back in range(1, 6):
                    j = cur_pos - back
                    if j < 0:
                        break
                    prev_doc = sortable[j][2]
                    prev_md = prev_doc.metadata or {}
                    prev_name = prev_md.get("company_name") or extract_company_name_from_text(prev_doc.content)
                    prev_page = prev_md.get("page")
                    # 같은 페이지 또는 인접 페이지면 신뢰
                    if prev_name and isinstance(prev_page, int) and abs(prev_page - page) <= 1:
                        return doc, md, str(prev_name)

        return doc, md, "unknown"

    def _merge_unknown_profiles(self, profiles: List[StartupProfile]) -> List[StartupProfile]:
        """
        unknown profile을 기존 회사에 병합 시도.

        원칙:
        - page 인접성(같은/인접 페이지) + 섹션 연속성(3/4/5 섹션 위주)으로 병합 후보를 선택
        """
        unknowns = [p for p in profiles if p.company_name.strip().lower() == "unknown"]
        knowns = [p for p in profiles if p.company_name.strip().lower() != "unknown"]
        if not unknowns:
            return profiles

        merged_knowns = knowns[:]

        for u in unknowns:
            target = None
            u_text = (u.merged_raw_text or "").strip()

            def looks_like_continuation_profile(text: str) -> bool:
                # 회사 개요 없이 3/4/5 섹션만 있는 경우가 대표적
                return bool(
                    re.match(r"^\s*(3|4|5)\.\s", text)
                    or re.search(r"\b(기술적\s*강점\s*및\s*한계|경쟁사\s*대비\s*차별성|실적)\b", text)
                )

            # 1) page 인접성으로 병합 후보 선택 (same/adjacent page)
            if u.pages:
                up_min, up_max = min(u.pages), max(u.pages)
                # 가장 가까운 known을 선택
                best = None
                best_dist = 10**9
                for k in merged_knowns:
                    if not k.pages:
                        continue
                    kp_min, kp_max = min(k.pages), max(k.pages)
                    dist = min(abs(kp_max - up_min), abs(up_max - kp_min), abs(kp_min - up_min))
                    if dist < best_dist:
                        best_dist = dist
                        best = k
                if best is not None and best_dist <= 1:
                    target = best

            # 2) 섹션 연속성이 강하면(3/4/5) 인접한 회사로 병합 우선
            if target is not None and looks_like_continuation_profile(u_text):
                self._merge_profile_into(target, u)
            else:
                merged_knowns.append(u)

        return merged_knowns

    def _merge_profile_into(self, target: StartupProfile, src: StartupProfile) -> None:
        """
        병합 규칙:
        - pages: unique + 정렬
        - raw_texts: 중복 제거(순서 유지)
        - metadata_list: 모두 유지
        - merged_raw_text: 자연스럽게 이어붙이기
        - 필드 값: 빈 값보다 채워진 값 우선, 더 긴 값 우선
        """
        # pages
        target.pages = sorted(set((target.pages or []) + (src.pages or [])))

        # raw_texts + metadata_list를 가능한 한 정렬된 상태로 병합
        combined_texts = (target.raw_texts or []) + (src.raw_texts or [])
        combined_meta = (target.metadata_list or []) + (src.metadata_list or [])

        # 1) (meta, text) 길이가 맞으면 meta 기반 정렬 후 dedup
        paired = []
        if len(combined_texts) == len(combined_meta):
            for m, t in zip(combined_meta, combined_texts):
                paired.append((m or {}, t))

            def mkey(pair):
                m = pair[0]
                p = m.get("page")
                ci = m.get("chunk_index")
                p = p if isinstance(p, int) else 10**9
                ci = ci if isinstance(ci, int) else 10**9
                return (p, ci)

            paired.sort(key=mkey)
            seen = set()
            new_meta = []
            new_texts = []
            for m, t in paired:
                k = (t or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                new_meta.append(m)
                new_texts.append(t)
            target.metadata_list = new_meta
            target.raw_texts = new_texts
        else:
            # 2) 길이가 다르면 text만 dedup (순서 유지), meta는 모두 유지
            seen = set()
            new_texts = []
            for t in combined_texts:
                k = (t or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                new_texts.append(t)
            target.raw_texts = new_texts
            target.metadata_list = combined_meta

        # merged_raw_text
        target.merged_raw_text = "\n\n".join(target.raw_texts).strip()

        # score
        target.score = max(float(target.score), float(src.score))
        target.source_file = target.source_file or src.source_file

        # 파싱을 다시 돌려서 더 풍부한 값 반영
        overview, technology, strengths_limits, diff, perf, refs = parse_profile_sections(target.merged_raw_text)

        # references merge
        target.references = list(dict.fromkeys((target.references or []) + (refs or []) + (src.references or [])))

        # field-wise merge (빈 값보다 채워진 값 / 더 긴 값 우선)
        def pick(a: Optional[str], b: Optional[str]) -> Optional[str]:
            a = (a or "").strip()
            b = (b or "").strip()
            if not a and b:
                return b
            if not b and a:
                return a
            return b if len(b) > len(a) else a

        target.company_overview.representative = pick(target.company_overview.representative, overview.representative)
        target.company_overview.founded_year = pick(target.company_overview.founded_year, overview.founded_year)
        target.company_overview.region = pick(target.company_overview.region, overview.region)
        target.company_overview.industry = pick(target.company_overview.industry, overview.industry)
        target.company_overview.main_product = pick(target.company_overview.main_product, overview.main_product)
        target.company_overview.venture_type = pick(target.company_overview.venture_type, overview.venture_type)
        target.company_overview.investment_status = pick(
            target.company_overview.investment_status, overview.investment_status
        )
        target.company_overview.website = pick(target.company_overview.website, overview.website)

        target.technology.summary = pick(target.technology.summary, technology.summary)
        target.technology.maturity = pick(target.technology.maturity, technology.maturity)
        target.technology.patent_ip = pick(target.technology.patent_ip, technology.patent_ip)

        # strengths/limitations: 더 많은 항목 우선
        if len(strengths_limits.strengths) > len(target.strengths_and_limitations.strengths):
            target.strengths_and_limitations.strengths = strengths_limits.strengths
        if len(strengths_limits.limitations) > len(target.strengths_and_limitations.limitations):
            target.strengths_and_limitations.limitations = strengths_limits.limitations

        target.differentiation.description = pick(target.differentiation.description, diff.description)
        if len(diff.core_advantages) > len(target.differentiation.core_advantages):
            target.differentiation.core_advantages = diff.core_advantages

        target.performance.customers_references = pick(target.performance.customers_references, perf.customers_references)
        target.performance.sales_growth = pick(target.performance.sales_growth, perf.sales_growth)
        target.performance.adoption_cases = pick(target.performance.adoption_cases, perf.adoption_cases)
