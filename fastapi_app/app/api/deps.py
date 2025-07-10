"""
API 엔드포인트에서 사용되는 의존성 모듈

이 모듈은 FastAPI 엔드포인트에서 사용되는 다양한 의존성들을 정의합니다.
주요 구성요소:
    - get_llm: LangChain LLM 의존성
    - get_recommender: 추천 서비스 의존성
    - get_place_store: 장소 벡터 저장소 의존성
    # TODO: 아래 의존성들은 추후 구현 예정
    # - get_logger: 로깅 의존성
    # - get_cache: 캐시 의존성
"""

import logging

from fastapi import Depends, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.services.recommend.service import RecommenderService
from app.services.recommend.keyword_extractor import KeywordExtractor
from app.services.llm_factory import LLMFactory
from app.services.recommend.retriever import PlaceStore
from app.services.place_store_factory import PlaceStoreFactory
from app.services.recommend.embedding import EmbeddingModel
from app.services.embedding_factory import EmbeddingModelFactory
from app.services.recommend.engine import RecommendationEngine
from app.logging.di import get_logger_dep
from monitoring.metrics import metrics as recommend_metrics  # 추천 API 메트릭 싱글턴 인스턴스 임포트
from app.services.moment.generator import GeneratorService
from app.data_pipeline.pipeline import UploaderPipeline
# TODO: 추후 구현 예정
# import logging
# from typing import Generator
# from app.logging.config import setup_logger
# from app.cache.redis import get_redis_client

# LLM 의존성
def get_llm() -> ChatGoogleGenerativeAI:
    """
    LangChain LLM 의존성
    
    Returns:
        ChatGoogleGenerativeAI: Gemini LLM 인스턴스
        
    Raises:
        HTTPException: LLM 초기화 실패 시
    """
    try:
        return LLMFactory.get_instance()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 초기화 실패: {str(e)}"
        )
    
def get_keyword_extractor(
    llm: ChatGoogleGenerativeAI = Depends(get_llm)
) -> KeywordExtractor:
    try:
        return KeywordExtractor(
            llm=llm
        )
    except Exception as e:
        logger.error(f"키워드 추출기 초기화 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"키워드 추출기 초기화 실패: {str(e)}"
        )

def get_embedding_model() -> EmbeddingModel:
    """
    임베딩 모델의 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        SentenceTransformer: 임베딩 모델 인스턴스
    """
    try:
        return EmbeddingModelFactory.get_instance()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Embedding model 초기화 실패: {str(e)}"
        )

def get_place_store() -> PlaceStore:
    try:
        return PlaceStoreFactory.get_instance()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PlaceStore 로딩 실패: {str(e)}"
        )

def get_recommendation_engine(
        place_store: PlaceStore = Depends(get_place_store)
) -> RecommendationEngine:
    try:
        return RecommendationEngine(
            place_store=place_store
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RecommendationEngine 초기화 실패: {str(e)}"
        )

# 추천 서비스 의존성
def get_recommender(
    keyword_extractor: KeywordExtractor = Depends(get_keyword_extractor),
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine),
    logger: logging.Logger = Depends(get_logger_dep)
) -> RecommenderService:
    """
    추천 서비스 의존성
    
    Args:
        llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        place_store (PlaceStore): 장소 벡터 저장소 인스턴스
        
    Returns:
        RecommenderService: 추천 서비스 인스턴스
        
    Raises:
        HTTPException: 서비스 초기화 실패 시
    """
    try:
        return RecommenderService(
            keyword_extractor=keyword_extractor,
            embedding_model=embedding_model,
            recommendation_engine=recommendation_engine
        )
    except Exception as e:
        logger.error(f"추천 서비스 초기화 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"추천 서비스 초기화 실패: {str(e)}"
        )

# 추천 API 메트릭을 의존성 주입으로 반환하는 함수
# FastAPI 엔드포인트에서 Depends(get_recommend_metrics)로 사용 가능
def get_recommend_metrics():
    return recommend_metrics

# 게시글 생성 서비스 의존성
def get_moment_generator(
    llm: ChatGoogleGenerativeAI = Depends(get_llm),
    logger: logging.Logger = Depends(get_logger_dep)
) -> GeneratorService:
    """
    게시글 생성 서비스 의존성
    
    Args:
        llm (ChatGoogleGenerativeAI): LangChain LLM 인스턴스
        
    Returns:
        GeneratorService: 게시글 생성 서비스 인스턴스
        
    Raises:
        HTTPException: 서비스 초기화 실패 시
    """
    try:
        return GeneratorService(
            llm=llm
        )
    except Exception as e:
        logger.error(f"게시글 생성 서비스 초기화 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"게시글 생성 서비스 초기화 실패: {str(e)}"
        )

# 데이터 업로더 의존성
def get_data_uploader(
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
    logger: logging.Logger = Depends(get_logger_dep)
) -> UploaderPipeline:
    try:
        return UploaderPipeline(
            embedding_model=embedding_model
        )
    except Exception as e:
        logger.error(f"데이터 업로더 초기화 실패: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"데이터 업로더 초기화 실패: {str(e)}"
        )
    
# TODO: 추후 구현 예정
# # 로깅 의존성
# def get_logger_dep() -> logging.Logger:
#     """
#     FastAPI 의존성 주입용 로거 반환 함수
#     """
#     return get_logger()

# # 캐시 의존성
# def get_cache():
#     """
#     캐시 의존성
#     
#     Returns:
#         Redis: Redis 클라이언트 인스턴스
#         
#     Raises:
#         HTTPException: 캐시 연결 실패 시
#     """
#     try:
#         return get_redis_client()
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"캐시 연결 실패: {str(e)}"
#         )
