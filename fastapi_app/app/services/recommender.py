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
            다음 사용자 입력에서 장소 추천에 관련된 키워드를 추출해주세요.
            키워드는 쉼표로 구분하여 반환해주세요.
            
            사용자 입력: {user_input}
            
            키워드:
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
