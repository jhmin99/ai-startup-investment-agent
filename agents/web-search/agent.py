"""
웹 검색 에이전트 - 단일 노드로 캡슐화

상위 그래프(Supervisor)에서 하나의 노드로 사용 가능:
    web_search_agent = WebSearchAgent()
    workflow.add_node("web_search", web_search_agent)
"""
import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from state import SearchResult
from tools import TavilySearchTool
from prompts import SEARCH_QUERY_GENERATION_PROMPT, FINAL_SUMMARY_PROMPT

load_dotenv()


# ============================================================
# 상수 정의
# ============================================================

CATEGORIES = ["market", "technology", "competition", "performance"]

CATEGORY_WEIGHTS = {
    "market": 35,       # 시장성
    "technology": 25,   # 기술력
    "competition": 20,  # 경쟁력
    "performance": 20,  # 실적
}

DEFAULT_MODEL = "gpt-4o"
MAX_SEARCH_RESULTS = 3
MAX_NEWS_RESULTS = 2


# ============================================================
# 웹 검색 에이전트
# ============================================================

class WebSearchAgent:
    """
    스타트업 투자 판단을 위한 웹 검색 에이전트
    
    단일 노드로 캡슐화되어 상위 그래프에서 사용 가능.
    내부적으로 쿼리 생성 → 카테고리별 검색 → 요약 생성 수행.
    """
    
    # --------------------------------------------------------
    # 초기화
    # --------------------------------------------------------
    
    def __init__(
        self,
        openai_api_key: str = None,
        tavily_api_key: str = None,
        model: str = DEFAULT_MODEL
    ):
        """
        Args:
            openai_api_key: OpenAI API 키 (없으면 환경변수 사용)
            tavily_api_key: Tavily API 키 (없으면 환경변수 사용)
            model: 사용할 LLM 모델
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.search_tool = TavilySearchTool(
            api_key=tavily_api_key or os.getenv("TAVILY_API_KEY")
        )
    
    # --------------------------------------------------------
    # 공개 인터페이스
    # --------------------------------------------------------
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        상위 그래프의 노드로 호출될 때 실행
        
        Args:
            state: 상위 그래프 상태 (startup_name 필수)
            
        Returns:
            투자 판단 에이전트용 구조화된 검색 결과
        """
        startup_name = state.get("startup_name", "")
        
        # 1단계: 검색 쿼리 생성
        search_queries = self._generate_queries(startup_name)
        
        # 2단계: 카테고리별 검색
        category_results = self._search_all_categories(search_queries)
        
        # 3단계: 핵심 팩트 추출
        findings = self._extract_findings(startup_name, category_results)
        
        # 투자 판단 에이전트용 구조화된 출력
        return {
            "startup_name": startup_name,
            "market": findings.get("market", {}).get("findings", []),
            "technology": findings.get("technology", {}).get("findings", []),
            "competition": findings.get("competition", {}).get("findings", []),
            "performance": findings.get("performance", {}).get("findings", []),
            "messages": [{"role": "assistant", "content": f"웹 검색 완료: {startup_name}"}]
        }
    
    def run(self, startup_name: str) -> Dict[str, Any]:
        """단독 실행"""
        return self({"startup_name": startup_name})
    
    async def arun(self, startup_name: str) -> Dict[str, Any]:
        """비동기 실행"""
        return self({"startup_name": startup_name})
    
    # --------------------------------------------------------
    # 1단계: 쿼리 생성
    # --------------------------------------------------------
    
    def _generate_queries(self, startup_name: str) -> Dict[str, List[str]]:
        """LLM을 사용해 카테고리별 검색 쿼리 생성"""
        prompt = SEARCH_QUERY_GENERATION_PROMPT.format(startup_name=startup_name)
        response = self.llm.invoke(prompt)
        
        try:
            return self._parse_json_response(response.content)
        except (json.JSONDecodeError, ValueError):
            return self._get_default_queries(startup_name)
    
    def _parse_json_response(self, content: str) -> Dict[str, List[str]]:
        """LLM 응답에서 JSON 추출"""
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content.strip())
    
    def _get_default_queries(self, startup_name: str) -> Dict[str, List[str]]:
        """JSON 파싱 실패 시 기본 쿼리 반환"""
        return {
            "market": [f"{startup_name} 시장 규모", f"{startup_name} 산업 전망"],
            "technology": [f"{startup_name} 기술 특허", f"{startup_name} 핵심 기술"],
            "competition": [f"{startup_name} 경쟁사", f"{startup_name} 시장 점유율"],
            "performance": [f"{startup_name} 투자 유치", f"{startup_name} 매출"]
        }
    
    # --------------------------------------------------------
    # 2단계: 검색 수행
    # --------------------------------------------------------
    
    def _search_all_categories(
        self, 
        search_queries: Dict[str, List[str]]
    ) -> Dict[str, List[SearchResult]]:
        """모든 카테고리 검색 수행"""
        return {
            category: self._search_category(search_queries, category)
            for category in CATEGORIES
        }
    
    def _search_category(
        self,
        search_queries: Dict[str, List[str]],
        category: str
    ) -> List[SearchResult]:
        """단일 카테고리 검색"""
        queries = search_queries.get(category, [])
        results = []
        
        for query in queries:
            # 일반 검색 + 뉴스 검색
            search_results = self.search_tool.search(query, max_results=MAX_SEARCH_RESULTS)
            news_results = self.search_tool.search_news(query, max_results=MAX_NEWS_RESULTS)
            
            for r in search_results + news_results:
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    category=category
                ))
        
        return results
    
    # --------------------------------------------------------
    # 3단계: 핵심 팩트 추출
    # --------------------------------------------------------
    
    def _extract_findings(
        self,
        startup_name: str,
        category_results: Dict[str, List[SearchResult]]
    ) -> Dict[str, Any]:
        """검색 결과에서 투자 판단용 핵심 팩트 추출"""
        prompt = FINAL_SUMMARY_PROMPT.format(
            startup_name=startup_name,
            market_info=self._format_results(category_results["market"]),
            technology_info=self._format_results(category_results["technology"]),
            competition_info=self._format_results(category_results["competition"]),
            performance_info=self._format_results(category_results["performance"])
        )
        
        response = self.llm.invoke(prompt)
        
        try:
            return self._parse_json_response(response.content)
        except (json.JSONDecodeError, ValueError):
            # 파싱 실패 시 빈 findings 반환
            return {
                "market": {"findings": []},
                "technology": {"findings": []},
                "competition": {"findings": []},
                "performance": {"findings": []}
            }
    
    def _format_results(self, results: List[SearchResult]) -> str:
        """검색 결과를 프롬프트용 문자열로 포맷"""
        if not results:
            return "검색 결과 없음"
        
        formatted = []
        for r in results:
            content_preview = r['content'][:200] + "..." if len(r['content']) > 200 else r['content']
            formatted.append(f"- [{r['title']}]({r['url']})\n  {content_preview}")
        
        return "\n".join(formatted)
    
    # --------------------------------------------------------
    # 유틸리티
    # --------------------------------------------------------
    
    def get_visualization(self) -> str:
        """Mermaid 형식 시각화"""
        return """
graph TD
    A[시작] --> B[웹 검색 에이전트]
    B --> C[종료]
    
    subgraph B[웹 검색 에이전트]
        B1[쿼리 생성] --> B2[시장성 검색]
        B2 --> B3[기술력 검색]
        B3 --> B4[경쟁력 검색]
        B4 --> B5[실적 검색]
        B5 --> B6[종합 요약]
    end
"""
