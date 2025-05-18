"""
추천 서비스 모듈

이 모듈은 추천 서비스의 핵심 비즈니스 로직을 구현합니다.
LangChain을 사용하여 사용자 입력을 처리하고 추천을 생성합니다.

주요 구성요소:
    - RecommenderService: 추천 서비스 클래스
"""

import json

from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.vector_store import PlaceStore
from app.schemas.recommend_schema import RecommendResponse
from app.services.recommend_engine import RecommendationEngine

class RecommenderService:
    """
    추천 서비스 클래스
    
    이 클래스는 사용자 입력을 처리하고 추천을 생성하는 서비스를 제공합니다.
    LangChain을 사용하여 자연어 처리를 수행합니다.
    
    Attributes:
        llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        chain: 키워드 추출을 위한 LangChain 체인
        recommendation_engine (RecommendationEngine): 추천 엔진
    """
    
    def __init__(
        self,
        llm: ChatGoogleGenerativeAI,
        place_store: PlaceStore
    ):
        """
        RecommenderService 초기화
        
        Args:
            llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
            place_store (PlaceStore): 장소 벡터 저장소 인스턴스
        """
        self.llm = llm
        self.chain = self._create_chain()
        self.recommendation_engine = RecommendationEngine(place_store)
    
    def _create_chain(self):
        """
        키워드 추출을 위한 LangChain 체인 생성
        
        Returns:
            RunnablePassthrough: 생성된 LangChain 체인
        """
        prompt = ChatPromptTemplate.from_template("""
            당신은 장소 검색 키워드 추출 AI입니다.

            아래에 제시된 사용자의 요청에서 사용자가 원하는 장소의 특징을 잘 나타내는 키워드를 추출하세요.

            사용자 요청:
            {user_input}

            키워드는 다음 7개의 카테고리로 분류합니다:

            1. 음식/제품: 제품, 음식 키워드 혹은 해당 제품, 음식 대한 표현(예: '부드러운 고기', '싱싱한 해산물', '맛있는 커피', '삼겹살', '양고기')
            2. 분위기/공간: 인테리어, 조명, 소음, 청결 등 (예: '조용한 분위기', '깔끔한 공간')
            3. 서비스/직원: 직원 응대, 서비스 속도 등 (예: '친절함', '빠른 응대')
            4. 가격/가성비: 합리적 가격, 혜택 등 (예: '가격 만족도 높음', '가성비 좋음')
            5. 접근성/편의시설: 주차, 위치 접근성, 편의시설 등 (예: '넉넉한 주차', '역 근처 위치', '24시간', '와이파이')
            6. 방문 목적: 추천 대상, 상황 등 (예: '데이트 추천', '혼밥 추천')
            7. 장소 카테고리 : 장소를 분류할 수 있는 키워드(예 : '카페', '한식')

            키워드 추출 조건:
            1. 사용자 요청에 대해 카테고리별로 장소의 특징을 잘 설명하는 키워드를 추출하세요.
            2. 형용사 키워드는 명사형 어미로 변경해서 추출하세요 (예: 친절한 → 친절함, 조용한 → 조용함).
            3. 형용사와 명사를 활용하여 구체적인 키워드를 추출해주세요 (예. 맛있는 커피, 시원한 냉면)
            4. 동일하거나 유사한 의미의 표현은 하나로 통일해서 중복 없이 추출하세요.
            5. [음식점, 카페, 편의점, 영화관] 중에서만 음식/제품에 적합한 장소 카테고리 키워드를 선택

            출력 형식:
            {{
            "음식/제품": [],
            "분위기/공간": [],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": [],
            "방문 목적": [],
            "장소 카테고리" : []
            }}
            """)
        
        return prompt | self.llm
    
    async def get_recommendation(self, user_input: str) -> RecommendResponse:
        """
        사용자 입력에서 키워드를 추출하고 추천 결과를 생성합니다.
        
        이 메서드는 다음 단계로 동작합니다:
        1. LangChain을 사용하여 사용자 입력에서 키워드 추출
        2. 추출된 키워드를 기반으로 장소 추천
        
        Args:
            user_input (str): 사용자의 입력 텍스트
            
        Returns:
            RecommendResponse: 추천 결과
            
        Raises:
            Exception: 추천 생성 과정에서 오류가 발생한 경우
        """
        try:
            # 1. 키워드 추출
            response = await self.chain.ainvoke({"user_input": user_input})
            keywords_str = response.content
            
            # JSON 문자열에서 키워드 딕셔너리 추출
            start_idx = keywords_str.find("{")
            end_idx = keywords_str.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError("키워드 추출 실패: JSON 형식이 올바르지 않습니다.")
            
            keywords_json = keywords_str[start_idx:end_idx]
            keywords = json.loads(keywords_json)
            # 2. 추천 생성
            return self.recommendation_engine.get_recommendations(keywords)
            
        except Exception as e:
            raise Exception(f"추천 생성 중 오류 발생: {str(e)}")
