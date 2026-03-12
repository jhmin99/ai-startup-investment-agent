"""
기술 요약 에이전트 상태 정의

TechInput: 스타트업 탐색 에이전트로부터 받는 입력
TechSummaryOutput: 투자 판단 에이전트로 전달하는 출력
"""
from typing import TypedDict, List, Optional, Dict, Any


# ============================================================
# 입력 타입 (스타트업 탐색 에이전트 → 기술 요약 에이전트)
# ============================================================

class TechInput(TypedDict):
    """
    스타트업 탐색 에이전트로부터 받는 기술 정보
    
    예시:
    {
        "startup_name": "수퍼빈",
        "technology_info": {
            "core_technology": "AI 기반 폐기물 자동 분류 시스템",
            "patents": ["특허1", "특허2"],
            "tech_stack": ["컴퓨터 비전", "딥러닝"],
            "rd_info": "R&D 인력 20명, 연구소 보유"
        }
    }
    """
    startup_name: str
    technology_info: Dict[str, Any]


# ============================================================
# 출력 타입 (기술 요약 에이전트 → 투자 판단 에이전트)
# ============================================================

class TechSummaryOutput(TypedDict):
    """
    투자 판단 에이전트로 전달하는 기술 요약
    
    투자 판단에 필요한 핵심 정보만 구조화하여 전달.
    """
    startup_name: str
    
    # 기술 요약
    core_technology: str          # 핵심 기술 한 줄 요약
    tech_summary: str             # 기술 상세 요약 (2-3문장)
    
    # 투자 판단 요소
    tech_strengths: List[str]     # 기술적 강점
    tech_weaknesses: List[str]    # 기술적 약점/리스크
    tech_differentiation: str     # 기술 차별화 포인트
    
    # 정량 지표 (있는 경우)
    patent_count: Optional[int]   # 특허 수
    rd_team_size: Optional[str]   # R&D 팀 규모
