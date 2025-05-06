"""
추천 서비스 모듈

이 모듈은 추천 서비스의 핵심 비즈니스 로직을 구현합니다.
LangChain을 사용하여 사용자 입력을 처리하고 추천을 생성합니다.

주요 구성요소:
    - RecommenderService: 추천 서비스 클래스
"""

from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class RecommenderService:
    """
    추천 서비스 클래스
    
    이 클래스는 사용자 입력을 처리하고 추천을 생성하는 서비스를 제공합니다.
    LangChain을 사용하여 자연어 처리를 수행합니다.
    
    Attributes:
        llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        chain (LLMChain): 키워드 추출을 위한 LangChain 체인
    """
    
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        RecommenderService 초기화
        
        Args:
            llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        """
        self.llm = llm
        self.chain = self._create_chain()
    
    def _create_chain(self) -> LLMChain:
        """
        키워드 추출을 위한 LangChain 체인 생성
        
        Returns:
            LLMChain: 생성된 LangChain 체인
        """
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""
            당신은 장소 검색 키워드 추출 AI입니다.

            아래에 제시된 사용자의 요청에서 사용자가 원하는 장소의 특징을 잘 나타내는 키워드를 추출하세요.

            키워드 추출 조건:
            1. 사용자 요청에 대해 **카테고리별로 장소의 특징을 잘 설명하는 키워드**를 추출하세요.
            2. **형용사 키워드**는 **명사형 어미**로 변경해서 추출하세요 (예: 친절한 → 친절함, 조용한 → 조용함).
            3. **n-gram(최대 3-gram)**을 활용하여 표현을 다양화하세요.
            4. **동일하거나 유사한 의미의 표현은 하나로 통일**해서 중복 없이 추출하세요.
            5. **긍정적인 의미의 키워드**만 추출하세요.

            키워드는 다음 6개의 카테고리로 분류합니다:

            1. **음식/제품**: 음식의 맛, 신선도, 품질 등 (예: '부드러운 고기', '싱싱한 해산물')
            2. **분위기/공간**: 인테리어, 조명, 소음, 청결 등 (예: '조용한 분위기', '깔끔한 공간')
            3. **서비스/직원**: 직원 응대, 서비스 속도 등 (예: '친절함', '빠른 응대')
            4. **가격/가성비**: 합리적 가격, 혜택 등 (예: '가격 만족도 높음', '가성비 좋음')
            5. **접근성/편의시설**: 주차, 위치 접근성, 편의시설 등 (예: '넉넉한 주차', '역 근처 위치', '24시간', '와이파이')
            6. **방문 목적**: 추천 대상, 상황 등 (예: '데이트 추천', '혼밥 추천')

            예시 
            사용자 요청:"여러명이서 회식하기 좋고 주차장 넓은 해산물 집 추천해줘"

            → 출력:

            사용자 키워드:
            {{
            "음식/제품": ["해산물"],
            "분위기/공간": ["매장 넓음"],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": ["주차장 넓음"],
            "방문 목적": ["회식 추천"]
            }}

            사용자 요청:"카공하기 좋은 조용하고 넓은 카페 추천해줘"

            → 출력:

            사용자 키워드: 
            장소 키워드:
            {{
            "음식/제품": [],
            "분위기/공간": ["은은한 조명", "조용한 분위기", "넓은 매장"],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": ["와이파이", "콘센트 이용 가능"],
            "방문 목적": ["카공족 추천"]
            }}
                
            사용자 요청:
            {user_input}

            출력 형식:

            사용자 키워드:
            {{
            "음식/제품": [],
            "분위기/공간": [],
            "서비스/직원": [],
            "가격/가성비": [],
            "접근성/편의시설": [],
            "방문 목적": []
            }}
            """
        )
        
        return LLMChain(
            llm=self.llm,
            prompt=prompt
        )
    
    async def get_recommendation(self, user_input: str) -> List[Dict]:
        """
        사용자 입력에서 키워드를 추출하고 추천 결과를 생성합니다.
        
        이 메서드는 다음 단계로 동작합니다:
        1. LangChain을 사용하여 사용자 입력에서 키워드 추출
        2. 추출된 키워드를 사용하여 벡터 유사도 검색 수행 (TODO)
        3. 검색 결과를 기반으로 추천 목록 생성
        
        Args:
            user_input (str): 사용자의 입력 텍스트
            
        Returns:
            List[Dict]: 추천 결과 목록. 각 항목은 id와 similarity_score를 포함
            
        Raises:
            Exception: 추천 생성 과정에서 오류가 발생한 경우
        """
        try:
            # LangChain을 사용하여 키워드 추출
            keywords = await self.chain.arun(user_input=user_input)
            
            # TODO: 추출된 키워드를 사용하여 벡터 유사도 검색 수행
            # 임시로 하드코딩된 응답 반환
            return [
                {"id": 1, "similarity_score": 0.95},
                {"id": 21, "similarity_score": 0.95},
                {"id": 32, "similarity_score": 0.95}
            ]
        except Exception as e:
            raise Exception(f"추천 생성 중 오류 발생: {str(e)}")
