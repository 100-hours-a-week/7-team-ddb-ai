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

logger = logging.getLogger(__name__)
from fastapi import Depends, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Generator

from app.core.config import settings
from app.services.recommender import RecommenderService
from app.services.vector_store import PlaceStore
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

def get_place_store() -> Generator[PlaceStore, None, None]:
    store = None
    try:
        logger.info("🔧 PlaceStore 인스턴스 생성 시도")
        store = PlaceStore()
        logger.info("✅ PlaceStore 인스턴스 생성 성공")
        yield store
        logger.info("✅ PlaceStore yield 이후 로직 실행됨")
    except Exception as e:
        import traceback
        logger.error("❌ get_place_store() 예외 발생:\n" + traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"벡터 저장소 초기화 실패: {str(e)}"
        )
    finally:
        if store:
            logger.info("🧹 PlaceStore 연결 종료 시도")
            store.close()


# 추천 서비스 의존성
def get_recommender(
    llm: ChatGoogleGenerativeAI = Depends(get_llm),
    place_store: PlaceStore = Depends(get_place_store)
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
        raise HTTPException(
            status_code=500,
            detail=f"추천 서비스 초기화 실패: {str(e)}"
        )

# TODO: 추후 구현 예정
# # 로깅 의존성
# def get_logger() -> logging.Logger:
#     """
#     로깅 의존성
#     
#     Returns:
#         logging.Logger: 설정된 로거 인스턴스
#     """
#     return setup_logger()

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
