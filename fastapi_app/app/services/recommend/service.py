"""
추천 서비스 모듈

이 모듈은 추천 서비스의 핵심 비즈니스 로직을 구현합니다.
LangChain을 사용하여 사용자 입력을 처리하고 추천을 생성합니다.

주요 구성요소:
    - RecommenderService: 추천 서비스 클래스
"""

import json
import time
import asyncio

from typing import List, Dict
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.recommend.retriever import PlaceStore
from app.schemas.recommend_schema import RecommendResponse
from app.services.recommend.embedding import EmbeddingModel
from app.services.recommend.engine import RecommendationEngine
from app.services.recommend.keyword_extractor import KeywordExtractor
from app.logging.di import get_logger_dep
from monitoring.metrics import RecommendMetrics  # 추천 API 메트릭 클래스 임포트

class RecommenderService:
    """
    추천 서비스 클래스
    
    이 클래스는 사용자 입력을 처리하고 추천을 생성하는 서비스를 제공합니다.
    LangChain을 사용하여 자연어 처리를 수행합니다.
    
    Attributes:
        keyword_extractor (KeywordExtractor): 키워드 추출
        recommendation_engine (RecommendationEngine): 추천 엔진
        metrics (RecommendMetrics): Prometheus 메트릭 객체
    """
    
    def __init__(
        self,
        keyword_extractor: KeywordExtractor,
        embedding_model: EmbeddingModel,
        recommendation_engine: RecommendationEngine,
        metrics=None,
        logger=None
    ):
        """
        RecommenderService 초기화
        
        Args:
            llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
            place_store (PlaceStore): 장소 벡터 저장소 인스턴스
            metrics (RecommendMetrics): Prometheus 메트릭 객체
        """
        self.keyword_extractor = keyword_extractor
        self.embedding_model = embedding_model
        self.recommendation_engine = recommendation_engine
        self.metrics = metrics  # DI로 주입받은 메트릭 객체 저장
        if logger is None:
            logger = get_logger_dep()
        self.logger = logger
    
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
        start = time.time()
        if self.metrics:
            self.metrics.request_count.inc()  # 추천 API 호출 시 카운터 증가
        try:
            # 1. 키워드 추출
            self.logger.info(f"추천 요청 : user_input = {user_input}")
            parsed, categories, keywords, place_category = await self.keyword_extractor.extract(user_input)

            # 추출 키워드가 없을 시 장소 카테고리만 반환
            if categories == None and keywords == None:
                return RecommendResponse(recommendations=[], place_category=place_category)
            # 추출 키워드가 있을 시 추천 시작
            else:
                # 2. 키워드 임베딩
                keywords_vec = await asyncio.to_thread(self.embedding_model.encode, keywords)
                # 3. 장소 추천 시작
                self.logger.info(f"추천 시작 : 키워드={parsed}")
                return await self.recommendation_engine.get_recommendations(categories, keywords_vec, place_category)
                
        except Exception as e:
            raise Exception(f"추천 생성 중 오류 발생: {str(e)}")
        finally:
            if self.metrics:
                # 추천 API 처리 시간 기록 (Histogram)
                self.metrics.request_latency.observe(time.time() - start)
