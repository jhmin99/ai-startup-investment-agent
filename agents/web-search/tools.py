"""웹 검색 에이전트 도구"""
import os
from typing import List, Dict, Any
from tavily import TavilyClient


class TavilySearchTool:
    """Tavily 기반 웹 검색 도구"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다.")
        self.client = TavilyClient(api_key=self.api_key)
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: List[str] = None,
        exclude_domains: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Tavily 검색 실행
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            search_depth: 검색 깊이 ("basic" | "advanced")
            include_domains: 포함할 도메인 목록
            exclude_domains: 제외할 도메인 목록
            
        Returns:
            검색 결과 리스트
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains or [],
                exclude_domains=exclude_domains or []
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"검색 오류: {e}")
            return []
    
    def search_news(
        self,
        query: str,
        max_results: int = 5,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        최신 뉴스 검색
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            days: 검색 기간 (일)
            
        Returns:
            뉴스 검색 결과 리스트
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                topic="news",
                days=days
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "published_date": result.get("published_date", ""),
                    "score": result.get("score", 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"뉴스 검색 오류: {e}")
            return []
    
    def search_with_context(
        self,
        query: str,
        max_results: int = 5
    ) -> Dict[str, Any]:
        """
        컨텍스트가 포함된 검색 (요약 포함)
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            검색 결과와 AI 생성 답변
        """
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
            
            return {
                "answer": response.get("answer", ""),
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", "")
                    }
                    for r in response.get("results", [])
                ]
            }
            
        except Exception as e:
            print(f"컨텍스트 검색 오류: {e}")
            return {"answer": "", "results": []}


def create_search_tool(api_key: str = None) -> TavilySearchTool:
    """검색 도구 팩토리 함수"""
    return TavilySearchTool(api_key=api_key)
