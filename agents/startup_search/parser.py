from __future__ import annotations

import re
from typing import List, Tuple

from .schemas import (
    CompanyOverview,
    DifferentiationSection,
    PerformanceSection,
    StrengthsAndLimitations,
    TechnologySection,
)


SECTION_PATTERNS: List[Tuple[str, str]] = [
    ("company_overview", r"\b1\.\s*회사\s*개요\b"),
    ("technology", r"\b2\.\s*기술\s*요약\b"),
    ("strengths_limitations", r"\b3\.\s*기술적\s*강점\s*및\s*한계\b"),
    ("differentiation", r"\b4\.\s*경쟁사\s*대비\s*차별성\b"),
    # 일부 문서는 5번이 없을 수 있어 optional
    ("performance", r"\b5\.\s*실적\b"),
]


def parse_profile_sections(text: str):
    """
    문서 원문을 섹션별로 나눠 CompanyOverview/Technology/Strengths... 등을 채움.
    파싱은 휴리스틱(정규식/문자열) 기반이며, 실패해도 None/빈 값으로 유지.
    """
    normalized_text = normalize_text_for_parsing(text or "")
    sections = split_into_sections(normalized_text)

    overview = parse_company_overview(sections.get("company_overview", ""))
    technology = parse_technology(sections.get("technology", ""))
    strengths_limits = parse_strengths_and_limitations(sections.get("strengths_limitations", ""))
    diff = parse_differentiation(sections.get("differentiation", ""))
    # performance는 merged text 전체에서 섹션 블록을 먼저 안전하게 분리한 뒤 파싱
    perf_block = extract_performance_section_block(normalized_text)
    perf = parse_performance_block(perf_block)

    refs = extract_references(normalized_text)

    return overview, technology, strengths_limits, diff, perf, refs


def split_into_sections(text: str) -> dict:
    """
    섹션 헤더를 기준으로 text를 나눔.

    중요:
    - RAG chunk가 겹치거나 일부 섹션이 중복 포함될 수 있어, 각 섹션은 "첫 매치"가 아니라
      후보들 중 더 완결된(길이/태그 포함) 블록을 선택한다.
    """
    t = text or ""

    # 모든 섹션 마커를 수집 (finditer)
    markers: List[Tuple[int, str]] = []
    for key, pat in SECTION_PATTERNS:
        for m in re.finditer(pat, t):
            markers.append((m.start(), key))
    markers.sort()
    if not markers:
        return {}

    # 마커 기준으로 candidate slice 생성
    candidates: dict[str, List[str]] = {}
    for i, (start, key) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(t)
        block = t[start:end].strip()
        if not block:
            continue
        candidates.setdefault(key, []).append(block)

    # 섹션별로 "가장 완결된" block 선택
    out: dict[str, str] = {}
    for key, blocks in candidates.items():
        out[key] = _pick_best_section_block(key, blocks)
    return out


def _pick_best_section_block(key: str, blocks: List[str]) -> str:
    """
    섹션 후보들 중 best를 고른다.
    - 기본: 길이가 긴 block
    - 3번(강점/한계): [강점] & [한계]를 모두 포함한 block을 우선
    - 5번(실적): 고객/매출/도입 라벨을 많이 포함한 block을 우선
    """
    if not blocks:
        return ""

    def score(block: str) -> tuple:
        b = block or ""
        length = len(b)

        if key == "strengths_limitations":
            has_strength = bool(re.search(r"[\[\［\【]\s*강점\s*[\]\］\】]", b))
            has_limit = bool(re.search(r"[\[\［\【]\s*한계\s*[\]\］\】]", b))
            tag_score = (1 if has_strength else 0) + (1 if has_limit else 0)
            return (tag_score, length)

        if key == "performance":
            label_cnt = 0
            for lab in ("고객/레퍼런스", "고객", "매출/성장", "매출", "도입 실적", "도입"):
                if lab in b:
                    label_cnt += 1
            return (label_cnt, length)

        return (0, length)

    return max(blocks, key=score)


def _extract_line_value(block: str, field_name: str) -> str | None:
    """
    "대표자 조**" 같이 '필드명 값' 형태를 잡아냄.
    """
    if not block:
        return None
    m = re.search(rf"{re.escape(field_name)}\s*[:：]?\s*([^\n]+)", block)
    if not m:
        return None
    return m.group(1).strip()


def normalize_text_for_parsing(text: str) -> str:
    """
    merged_raw_text에서 반복 헤더/페이지 노이즈를 제거해 파싱 안정성 향상.

    제거 대상(예시):
    - '로봇 스타트업 기업 정보 데이터셋'
    - '-- 92 of 155 --' 같은 페이지 마커
    - 숫자만 있는 페이지 번호 라인
    """
    if not text:
        return ""

    cleaned: List[str] = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        if l in {"로봇 스타트업 기업 정보 데이터셋", "로봇 스타트업 기업 정보", "데이터셋"}:
            continue
        if re.match(r"^--\s*\d+\s+of\s+\d+\s*--$", l):
            continue
        if re.match(r"^\d{1,3}$", l):
            # 페이지 번호로 추정되는 단독 숫자
            continue
        cleaned.append(line.rstrip())

    # 과도한 빈 줄 정리
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def _join_wrapped_lines(text: str) -> str:
    """줄바꿈으로 끊긴 문장을 최소한으로 join (의미 해석 없이 공백만 정리)."""
    if not text:
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _extract_labeled_block(
    block: str, label: str, stop_labels: List[str] | None = None, stop_section: bool = True
) -> str:
    """
    라벨 이후의 값을 '다음 라벨/다음 섹션 전까지' 멀티라인으로 추출.

    예: '투자 현황 ... (여러 줄) ... 웹사이트 ...' 에서 투자 현황을 웹사이트 전까지
    """
    if not block:
        return ""
    stop_labels = stop_labels or []

    label_pat = rf"{re.escape(label)}\s*[:：]?\s*"
    m = re.search(label_pat, block)
    if not m:
        return ""
    start = m.end()
    rest = block[start:]

    stops: List[str] = []
    for sl in stop_labels:
        stops.append(rf"{re.escape(sl)}\s*[:：]?\s*")
    if stop_section:
        # 다음 번호 섹션(예: 2. 기술 요약) 또는 문서 끝
        stops.append(r"\n\s*\d+\.\s*[^\n]+")

    if not stops:
        return rest.strip()

    stop_pat = "(" + "|".join(stops) + ")"
    m2 = re.search(stop_pat, rest)
    if m2:
        rest = rest[: m2.start()]
    return rest.strip()


def parse_company_overview(section_text: str) -> CompanyOverview:
    s = section_text or ""
    return CompanyOverview(
        representative=_extract_line_value(s, "대표자"),
        founded_year=_extract_line_value(s, "설립 연도"),
        region=_extract_line_value(s, "지역"),
        industry=_extract_line_value(s, "업종"),
        main_product=_extract_line_value(s, "주생산품"),
        venture_type=_extract_line_value(s, "벤처 유형"),
        # 투자 현황은 웹사이트 라벨 전까지 멀티라인 캡처
        investment_status=_join_wrapped_lines(
            _extract_labeled_block(s, "투자 현황", stop_labels=["웹사이트"], stop_section=False)
        )
        or None,
        website=_extract_line_value(s, "웹사이트"),
    )


def parse_technology(section_text: str) -> TechnologySection:
    s = section_text or ""
    # "2. 기술 요약" 이하 본문을 summary로 유지 (의미 해석 X)
    summary = _strip_section_header(s)
    return TechnologySection(
        summary=summary or None,
        # 기술 성숙도는 특허/IP 또는 다음 섹션 전까지 멀티라인 캡처
        maturity=_join_wrapped_lines(
            _extract_labeled_block(s, "기술 성숙도", stop_labels=["특허/IP"], stop_section=True)
        )
        or None,
        # 특허/IP는 다음 섹션 전까지
        patent_ip=_join_wrapped_lines(
            _extract_labeled_block(s, "특허/IP", stop_labels=[], stop_section=True)
        )
        or None,
    )


def parse_strengths_and_limitations(section_text: str) -> StrengthsAndLimitations:
    s = section_text or ""
    # [강점] ... [한계] ...
    # 강점은 [한계] 전까지가 최우선 경계 (줄바꿈 없이 붙는 케이스 포함)
    strengths_block = _extract_bracket_block(
        s,
        "강점",
        stop_at_section=True,
        extra_stop_patterns=[r"[\[\［\【]\s*한계\s*[\]\］\】]"],
    )
    limitations_block = _extract_bracket_block(s, "한계", stop_at_section=True)
    return StrengthsAndLimitations(
        strengths=_split_block_items(strengths_block),
        limitations=_split_block_items(limitations_block),
    )


def parse_differentiation(section_text: str) -> DifferentiationSection:
    s = section_text or ""
    # description은 [핵심 경쟁 우위] 또는 다음 섹션 전까지
    desc = _strip_section_header(s)
    if desc:
        m = re.search(r"\n\s*\[핵심\s*경쟁\s*우위\]\s*", desc)
        if m:
            desc = desc[: m.start()].strip()
    description = desc or None

    # 핵심 경쟁 우위는 5. 실적 또는 다음 섹션 전까지
    core_adv = _extract_bracket_block(s, "핵심 경쟁 우위", stop_at_section=True, extra_stop_patterns=[r"\n\s*5\.\s*실적"])
    core_advantages = _split_block_items(core_adv)

    return DifferentiationSection(
        description=description,
        core_advantages=core_advantages,
    )


def extract_performance_section_block(text: str) -> str:
    """
    merged text에서 '5. 실적' 섹션 블록만 안전하게 잘라냄.

    끝 경계:
    - 다음 번호 섹션 (6. ...)
    - 다음 회사 헤더 패턴 (예: '46. 주식회사 토트', '(주)...', '주식회사 ...')
    - 문서 종료
    """
    t = text or ""
    # 회사 본문 중간에 나오는 '5. 실적'을 찾는다 (첫 번째 매치 사용)
    m = re.search(r"(?:^|\n)\s*5\.\s*실적[^\n]*", t)
    if not m:
        return ""
    start = m.start()
    rest = t[m.end() :]

    # 다음 섹션(6.) 또는 다음 회사 헤더로 종료
    stop_pat = (
        r"(\n\s*6\.\s*[^\n]+)"  # 다음 번호 섹션
        r"|(\n\s*\d+\.\s*(?:주식회사|\(주\)|\([^)]*\)|[^\n]{2,80}))"  # 다음 회사 헤더
    )
    m2 = re.search(stop_pat, rest)
    end = m.end() + (m2.start() if m2 else len(rest))
    return t[start:end].strip()


def parse_performance_block(perf_block: str) -> PerformanceSection:
    """
    performance 블록 안에서만 고객/레퍼런스, 매출/성장, 도입 실적을 분리.
    """
    if not perf_block:
        return PerformanceSection()

    body = _strip_section_header(perf_block)
    # 제목 라인 제거
    body = re.sub(r"^\s*5\.\s*실적[^\n]*\n?", "", body).strip()

    if not body:
        return PerformanceSection()

    # 라벨 위치 기반 분리 (라벨이 같은 줄에 붙는 케이스까지 커버)
    # - 라벨은 줄 시작에 없어도 되며, 콜론(:) 유무도 허용
    # - 여러 라벨이 한 줄에 연속으로 있어도, "라벨 위치" 기준으로 다음 라벨 전까지만 값으로 취급
    label_variants = {
        "customers_references": ["고객/레퍼런스", "고객", "고객사"],
        "sales_growth": ["매출/성장", "매출 성장", "매출"],
        "adoption_cases": ["도입 실적", "도입"],
    }

    def find_first_span(key: str) -> tuple[int, int] | None:
        for lab in label_variants[key]:
            m = re.search(rf"{re.escape(lab)}\s*[:：]?\s*", body)
            if m:
                return (m.start(), m.end())
        return None

    spans: List[tuple[int, int, str]] = []
    for k in ("customers_references", "sales_growth", "adoption_cases"):
        sp = find_first_span(k)
        if sp:
            spans.append((sp[0], sp[1], k))
    spans.sort(key=lambda x: x[0])

    extracted = {"customers_references": None, "sales_growth": None, "adoption_cases": None}
    if spans:
        for i, (st, ed, key) in enumerate(spans):
            nxt = spans[i + 1][0] if i + 1 < len(spans) else len(body)
            val = body[ed:nxt].strip()

            # 값에서 노이즈 제거/정규화
            val = normalize_text_for_parsing(val)
            val = _join_wrapped_lines(val).strip()

            # 값에 다른 라벨 토큰이 섞여 들어오면(줄바꿈/구분자 깨짐) 그 지점에서 컷
            other_labels = []
            for kk, labs in label_variants.items():
                if kk == key:
                    continue
                other_labels.extend(labs)
            for lab in sorted(other_labels, key=len, reverse=True):
                mcut = re.search(rf"\b{re.escape(lab)}\b", val)
                if mcut:
                    val = val[: mcut.start()].strip()

            extracted[key] = val or None

    customers = extracted["customers_references"]
    sales = extracted["sales_growth"]
    adoption = extracted["adoption_cases"]

    # 참고/링크 블록이 adoption에 섞이면 컷 (references는 별도 추출됨)
    if adoption:
        mref = re.search(r"\b참고\s*[:：]", adoption)
        if mref:
            adoption = adoption[: mref.start()].strip()

    # '정보 없음' 정규화
    if customers and re.search(r"\b정보\s*없음\b", customers):
        customers = "정보 없음"
    # 매출이 '정보 없음'으로 명시된 경우 문자열로 통일
    if sales:
        if re.search(r"\b정보\s*없음\b", sales):
            sales = "정보 없음"
    # 라벨은 있는데 값이 없으면 '정보 없음'으로
    if sales is None and any(k == "sales_growth" for _, _, k in spans):
        sales = "정보 없음"

    # 값이 없으면 None으로 통일
    customers = customers or None
    sales = sales or None
    adoption = adoption or None

    return PerformanceSection(
        customers_references=customers,
        sales_growth=sales,
        adoption_cases=adoption,
    )


def extract_references(text: str) -> List[str]:
    """원문에서 URL을 모두 추출."""
    urls = re.findall(r"https?://[^\s\]\)]+", text or "")
    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _strip_section_header(section_text: str) -> str:
    """'2. 기술 요약' 같은 첫 줄 헤더를 제거하고 본문을 최대한 원문 그대로 반환."""
    if not section_text:
        return ""
    lines = [l.rstrip() for l in section_text.splitlines()]
    if not lines:
        return ""
    # 헤더처럼 보이는 첫 줄 제거
    if re.match(r"^\s*\d+\.\s*", lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _extract_bracket_block(
    text: str, label: str, stop_at_section: bool = False, extra_stop_patterns: List[str] | None = None
) -> str:
    """
    [강점] ... 다음 [..] 또는 섹션 끝까지 추출.
    """
    if not text:
        return ""
    extra_stop_patterns = extra_stop_patterns or []
    # 라벨 표기 변형 지원: [강점], ［강점］, 【강점】 등
    open_b = r"[\[\［\【]"
    close_b = r"[\]\］\】]"
    m = re.search(rf"{open_b}\s*{re.escape(label)}\s*{close_b}\s*(.*)", text, flags=re.DOTALL)
    if not m:
        return ""
    rest = m.group(1)
    # 다음 태그/섹션/회사 헤더로 안전하게 stop (줄바꿈 없이 붙는 태그도 처리)
    stop_candidates = [
        r"[\[\［\【]\s*[^\]\］\】]+\s*[\]\］\】]",  # 다음 [태그] (newline 없어도, 변형 포함)
        r"(?:^|\n)\s*\d+\.\s*(?:주식회사|\(주\)|\([^)]*\)|[^\n]{2,80})",  # 다음 회사 헤더
    ] + extra_stop_patterns
    if stop_at_section:
        stop_candidates.append(r"(?:^|\n)\s*\d+\.\s*[^\n]+")

    # 강점 블록에 기술요약 라벨이 섞여 들어오는 것을 방지 (maturity/patent 오염 차단)
    if label == "강점":
        stop_candidates.extend(
            [
                r"(?:^|\n)\s*기술\s*성숙도\s*[:：]?\s*",
                r"(?:^|\n)\s*특허\s*/\s*IP\s*[:：]?\s*",
                r"(?:^|\n)\s*특허\s*IP\s*[:：]?\s*",
            ]
        )
    m2 = re.search("(" + "|".join(stop_candidates) + ")", rest)
    if m2:
        rest = rest[: m2.start()]
    return rest.strip()


def _split_block_items(block: str) -> List[str]:
    """
    강점/한계/핵심 경쟁 우위 블록을 안정적으로 리스트화.

    전략:
    1) 먼저 줄바꿈으로 끊긴 문장을 join (중간 잘림 방지)
    2) bullet 형태가 명확하면 bullet 단위로
    3) 문장 분리가 애매하면 블록 전체를 1개 원소로 유지
    """
    raw = (block or "").strip()
    if not raw:
        return []

    def looks_truncated_tail(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        # 종결부호가 없고 짧은 경우는 대개 청크 절단 조각
        if len(s) <= 35 and not re.search(r"[\.\!\?]$", s):
            return True
        # 명백히 미완성 어절/접속어/조사로 끝나는 경우(예시 기반 + 일반화)
        if re.search(r"(전문|부품을|자율|있습니)$", s):
            return True
        if re.search(r"(그리고|또한|특히|다만|또는)$", s):
            return True
        if re.search(r"(은|는|이|가|을|를|에|의|과|와|로|으로|및)$", s):
            return True
        return False

    # 줄 단위로 봤을 때 마지막 라인이 잘린 조각이면 앞 라인에 우선 결합
    raw_lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if len(raw_lines) >= 2 and looks_truncated_tail(raw_lines[-1]):
        raw_lines[-2] = (raw_lines[-2].rstrip() + " " + raw_lines[-1].lstrip()).strip()
        raw_lines = raw_lines[:-1]
        raw = "\n".join(raw_lines).strip()

    # bullet 라인이 명확한 경우: 각 bullet 내부는 wrapped join
    bullet_lines = [l for l in raw.splitlines() if re.match(r"^\s*[\-\*\•]\s+", l)]
    if len(bullet_lines) >= 2:
        items = []
        for l in bullet_lines:
            items.append(re.sub(r"^\s*[\-\*\•]\s+", "", l).strip())
        items = [re.sub(r"\s+", " ", i).strip() for i in items if i.strip()]
        return items if items else [_join_wrapped_lines(raw)]

    joined = _join_wrapped_lines(raw)
    if not joined:
        return []

    # 너무 짧으면 쪼개지 말고 그대로
    if len(joined) < 80:
        return [joined]

    # 마침표/물음표/느낌표 기반으로만 최소 분리 (한국어는 종결부호가 없는 경우가 많아 과분리 방지)
    parts = [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+", joined) if p.strip()]

    if not parts:
        return [joined]

    # 청크 끝에서 잘린 조각을 앞 문장에 붙이기
    def is_truncated_fragment(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # 종결 부호 없고 너무 짧은 경우
        if len(s) <= 25 and not re.search(r"[\.\!\?]$", s):
            return True
        # 명백히 미완성 어절로 끝나는 경우
        bad_endings = ("전문", "부품을", "자율", "있습니")
        if any(s.endswith(be) for be in bad_endings):
            return True
        return False

    fixed: List[str] = []
    for p in parts:
        if fixed and is_truncated_fragment(p):
            fixed[-1] = (fixed[-1].rstrip() + " " + p.lstrip()).strip()
        else:
            fixed.append(p)

    # 여전히 분리 품질이 낮으면(짧은 조각이 많으면) 덩어리 1개로 유지
    short_cnt = sum(1 for x in fixed if len(x) < 20)
    if short_cnt >= 2 and len(fixed) >= 3:
        return [joined]

    # 단일 항목인데 꼬리가 명백히 잘렸으면(검색 청크 절단) 표식을 남김
    if len(fixed) == 1 and looks_truncated_tail(fixed[0]):
        return [fixed[0].rstrip() + "..."]

    # 마지막 항목이 잘린 조각이면 앞 항목에 결합(분리된 경우)
    if len(fixed) >= 2 and looks_truncated_tail(fixed[-1]):
        fixed[-2] = (fixed[-2].rstrip() + " " + fixed[-1].lstrip()).strip()
        fixed = fixed[:-1]

    return fixed

