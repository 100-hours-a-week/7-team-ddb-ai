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
from typing import Generator

from app.core.config import settings
from app.services.recommender import RecommenderService
from app.services.vector_store import PlaceStore
from app.logging.di import get_logger_dep
from monitoring.metrics import metrics as recommend_metrics  # 추천 API 메트릭 싱글턴 인스턴스 임포트
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
        return ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 초기화 실패: {str(e)}"
        )

def get_place_store(logger: logging.Logger = Depends(get_logger_dep)) -> Generator[PlaceStore, None, None]:
    store = None
    try:
        store = PlaceStore()
        yield store
    except Exception as e:
        import traceback
        logger.error("❌ get_place_store() 예외 발생:\n" + traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"벡터 저장소 초기화 실패: {str(e)}"
        )
    finally:
        if store:
            store.close()


# 추천 서비스 의존성
def get_recommender(
    llm: ChatGoogleGenerativeAI = Depends(get_llm),
    place_store: PlaceStore = Depends(get_place_store),
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
            llm=llm,
            place_store=place_store
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
