from __future__ import annotations

from html import escape
from typing import Iterable

from .schemas import Reference


def safe_text(value: str | None, default: str = "미상") -> str:
    if value is None:
        return default
    stripped = str(value).strip()
    if not stripped:
        return default
    return escape(stripped)


def join_list(values: list[str], default: str = "미상") -> str:
    cleaned = [escape(v.strip()) for v in values if v and v.strip()]
    if not cleaned:
        return default
    return ", ".join(cleaned)


def render_tag_list(values: list[str], tag_class: str) -> str:
    if not values:
        return f'<span class="tag {tag_class}">없음</span>'
    return "\n".join(
        f'<span class="tag {tag_class}">{escape(value)}</span>' for value in values[:3]
    )


def render_bullet_list(values: list[str], default_message: str = "정보 없음") -> str:
    if not values:
        return f'<ul class="bullet-list"><li>{escape(default_message)}</li></ul>'
    items = "\n".join(f"<li>{escape(value)}</li>" for value in values)
    return f'<ul class="bullet-list">\n{items}\n</ul>'


def render_limit_chips(values: list[str]) -> str:
    if not values:
        values = ["공개 데이터만 사용", "시장 상황 변동 가능"]
    return "\n".join(
        f'<span class="limit-chip">{escape(value)}</span>' for value in values
    )


def render_references(references: Iterable[Reference]) -> str:
    refs = list(references)
    if not refs:
        return '<div class="ref-item"><span>📄</span>참고 자료 없음</div>'

    icon_map = {
        "rag": "📄",
        "web": "📰",
        "manual": "📝",
    }

    rendered = []
    for ref in refs[:6]:
        icon = icon_map.get(ref.source_type, "📄")
        label = escape(ref.title)
        if ref.url.strip():
            item = (
                f'<div class="ref-item"><span>{icon}</span>'
                f'<a href="{escape(ref.url)}" target="_blank" rel="noopener noreferrer">{label}</a>'
                f"</div>"
            )
        else:
            item = f'<div class="ref-item"><span>{icon}</span>{label}</div>'
        rendered.append(item)

    return "\n".join(rendered)


def score_to_30(score: int) -> int:
    return round(score / 5 * 30)


def score_to_25(score: int) -> int:
    return round(score / 5 * 25)


def score_to_20(score: int) -> int:
    return round(score / 5 * 20)


def verdict_label(verdict: str) -> str:
    mapping = {
        "적극 투자": "적극 투자",
        "투자 가능": "투자 가능",
        "보류": "보류",
        "투자 위험": "투자 위험",
    }
    return mapping.get(verdict, verdict)


def pass_fail_label(condition: bool) -> str:
    return "충족" if condition else "미충족"


def pass_fail_class(condition: bool) -> str:
    return "s-pass" if condition else "s-fail"


def html_to_pdf(html: str, output_path: str) -> None:
    """
    HTML 문자열을 PDF 파일로 저장.

    - 기본 구현은 weasyprint를 사용하며, 설치되어 있지 않으면 친절한 에러를 던진다.
    - 시스템에 weasyprint + Cairo 등이 안 깔려 있으면, 이 함수를 호출하는 쪽에서
      try/except로 잡고 "HTML만 사용"하는 fallback을 쓰면 된다.
    """
    try:
        from weasyprint import HTML  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "HTML을 PDF로 변환하려면 weasyprint가 필요합니다. "
            "예: pip install weasyprint"
        ) from exc

    HTML(string=html).write_pdf(output_path)

