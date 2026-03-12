"""
기술 요약 에이전트 - 단일 노드로 캡슐화

스타트업 탐색 에이전트로부터 기술 정보를 받아
투자 판단 에이전트가 활용할 수 있는 형태로 요약.

상위 그래프에서 사용:
    tech_summary_agent = TechSummaryAgent()
    workflow.add_node("tech_summary", tech_summary_agent)
"""
import os
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from state import TechInput, TechSummaryOutput
from prompts import TECH_SUMMARY_PROMPT

load_dotenv()


# ============================================================
# 상수 정의
# ============================================================

DEFAULT_MODEL = "gpt-4o"


# ============================================================
# 기술 요약 에이전트
# ============================================================

class TechSummaryAgent:
    """
    기술 요약 에이전트
    
    스타트업의 기술 정보를 분석하여 투자 판단에 필요한 형태로 요약.
    단일 노드로 캡슐화되어 상위 그래프에서 사용 가능.
    """
    
    # --------------------------------------------------------
    # 초기화
    # --------------------------------------------------------
    
    def __init__(
        self,
        openai_api_key: str = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Args:
            openai_api_key: OpenAI API 키 (없으면 환경변수 사용)
            model: 사용할 LLM 모델
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
    
    # --------------------------------------------------------
    # 공개 인터페이스
    # --------------------------------------------------------
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        상위 그래프의 노드로 호출될 때 실행
        
        Args:
            state: 상위 그래프 상태
                - startup_name: 스타트업 이름
                - technology_info: 기술 정보 (dict 또는 str)
            
        Returns:
            기술 요약 결과
        """
        startup_name = state.get("startup_name", "")
        technology_info = state.get("technology_info", {})
        
        # 기술 요약 생성
        summary = self._generate_summary(startup_name, technology_info)
        
        return {
            "startup_name": startup_name,
            **summary,
            "messages": [{"role": "assistant", "content": f"기술 요약 완료: {startup_name}"}]
        }
    
    def run(self, startup_name: str, technology_info: Dict[str, Any]) -> Dict[str, Any]:
        """단독 실행"""
        return self({
            "startup_name": startup_name,
            "technology_info": technology_info
        })
    
    def run_batch(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 스타트업 일괄 처리
        
        Args:
            inputs: [{"startup_name": "...", "technology_info": {...}}, ...]
        
        Returns:
            각 스타트업의 기술 요약 리스트
        """
        results = []
        for inp in inputs:
            result = self(inp)
            results.append(result)
        return results
    
    # --------------------------------------------------------
    # 기술 요약 생성
    # --------------------------------------------------------
    
    def _generate_summary(
        self,
        startup_name: str,
        technology_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """기술 정보를 분석하여 요약 생성"""
        
        # 기술 정보를 문자열로 변환
        if isinstance(technology_info, dict):
            tech_info_str = json.dumps(technology_info, ensure_ascii=False, indent=2)
        else:
            tech_info_str = str(technology_info)
        
        prompt = TECH_SUMMARY_PROMPT.format(
            startup_name=startup_name,
            technology_info=tech_info_str
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            return self._parse_json_response(response.content)
        except (json.JSONDecodeError, ValueError):
            # 파싱 실패 시 기본값 반환
            return self._get_default_summary(startup_name, technology_info)
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """LLM 응답에서 JSON 추출"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    
    def _get_default_summary(
        self,
        startup_name: str,
        technology_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """파싱 실패 시 기본 요약 반환"""
        return {
            "core_technology": technology_info.get("core_technology", "정보 없음"),
            "tech_summary": "기술 정보 분석 실패",
            "tech_strengths": [],
            "tech_weaknesses": [],
            "tech_differentiation": "정보 없음",
            "patent_count": None,
            "rd_team_size": None
        }
    
    # --------------------------------------------------------
    # 유틸리티
    # --------------------------------------------------------
    
    def get_visualization(self) -> str:
        """Mermaid 형식 시각화"""
        return """
graph TD
    A[스타트업 탐색 에이전트] --> B[기술 요약 에이전트]
    B --> C[투자 판단 에이전트]
    
    subgraph B[기술 요약 에이전트]
        B1[기술 정보 수신] --> B2[LLM 분석]
        B2 --> B3[구조화된 요약 생성]
    end
"""
