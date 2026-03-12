from langchain_core.messages import HumanMessage

from agents.investment_decision_agent.agent import investment_decision_node


def main() -> None:
    state = {
        "messages": [
            HumanMessage(content="Robotics 도메인의 AI 스타트업 투자 가능성을 평가해줘")
        ],
        "user_query": "Robotics 도메인의 AI 스타트업 투자 가능성을 평가해줘",
        # investment_decision_agent는 dict 기반 state도 허용
        "startup_profile": {
            "company_name": "RoboNext AI",
            "one_liner": "산업용 비전 기반 자율 피킹 로봇을 개발하는 로보틱스 스타트업",
            "headquarters": "Seoul, Korea",
            "founded_year": 2022,
            "representative": "홍길동",
            "products": ["3D Vision Picking System", "AI Robot Cell"],
            "customers": ["국내 2차전지 부품사 파일럿", "스마트팩토리 SI사"],
            "fundraising_history": ["Seed 투자 유치 완료", "TIPS 선정"],
            "patents": ["3D 비전 기반 그리퍼 제어 특허 출원"],
        },
        "exploration_summary": {
            "competitor_summary": (
                "기존 자동화 장비 업체 대비 비정형 부품 인식 정확도가 높고, 셀 단위 도입 비용이 낮습니다. "
                "다만 대규모 해외 레퍼런스는 아직 부족합니다."
            ),
            "traction_summary": (
                "국내 제조 고객사 대상으로 PoC를 수행했고 일부 유료 파일럿 전환이 진행 중입니다. "
                "초기 매출은 제한적이지만 TIPS와 시드 투자 이력이 존재합니다."
            ),
            "references": [
                {
                    "title": "벤처기업 확인 정보",
                    "url": "https://www.smes.go.kr/venturein/",
                    "source_type": "rag",
                }
            ],
        },
        "technology_summary": {
            "core_technology": "3D 비전, 딥러닝 기반 객체 인식, 그리퍼 제어 최적화",
            "strengths": [
                "비정형 부품 인식 정확도 향상",
                "기존 라인에 후행 설치 가능한 모듈형 구조",
                "비전 데이터 축적 기반 성능 개선 가능",
            ],
            "limitations": [
                "고객사 환경별 튜닝 비용 발생",
                "대규모 양산 레퍼런스 부족",
            ],
            "differentiation": "하드웨어와 비전 소프트웨어를 함께 제공해 구축 기간을 줄일 수 있습니다.",
            "technical_maturity": "파일럿-초기 상용화 단계",
            "references": [
                {
                    "title": "회사 소개서",
                    "url": "https://example.com/company-deck",
                    "source_type": "manual",
                }
            ],
        },
        "market_assessment": {
            "market_size": "스마트팩토리 및 물류 자동화 시장 중심으로 중장기 확장 가능성이 높습니다.",
            "growth_drivers": [
                "제조업 인력 부족",
                "품질 자동화 수요 증가",
                "AI 비전 기술 상용화 가속",
            ],
            "target_industries": ["2차전지", "전자부품", "물류 자동화"],
            "scalability": "반복 공정이 많은 제조 분야에서 고객 확장이 가능하며, 향후 물류 영역으로도 수평 확장 여지가 있습니다.",
            "commercialization_risk": "현장 튜닝과 설치 리드타임이 길어질 경우 매출 인식이 지연될 수 있습니다.",
            "references": [
                {
                    "title": "로봇산업 실태조사",
                    "url": "https://www.data.go.kr/",
                    "source_type": "rag",
                }
            ],
        },
    }

    result = investment_decision_node(state)
    decision = result["investment_decision"]

    print("=" * 80)
    print("[투자 판단 Agent 결과]")
    print(f"시장성 점수: {decision.scorecard.market.score}/5")
    print(f"기술력 점수: {decision.scorecard.technology.score}/5")
    print(f"경쟁력 점수: {decision.scorecard.competitiveness.score}/5")
    print(f"실적 점수: {decision.scorecard.traction.score}/5")
    print("-" * 80)
    print(f"시장성 가중 점수: {decision.weighted_score.market}")
    print(f"기술력 가중 점수: {decision.weighted_score.technology}")
    print(f"경쟁력 가중 점수: {decision.weighted_score.competitiveness}")
    print(f"실적 가중 점수: {decision.weighted_score.traction}")
    print(f"총점: {decision.weighted_score.total}")
    print(f"최종 판단: {decision.verdict}")
    print(f"웹 검색 필요 여부: {decision.requires_web_search}")
    print(f"보완 필요 정보: {decision.missing_information}")
    print("-" * 80)
    print("[판단 근거]")
    for item in decision.decision_rationale:
        print(f"- {item}")
    print("=" * 80)


if __name__ == "__main__":
    main()
