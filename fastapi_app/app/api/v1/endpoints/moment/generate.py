"""
추천 API 엔드포인트

이 모듈은 추천 서비스의 REST API 엔드포인트를 제공합니다.
FastAPI를 사용하여 HTTP 요청을 처리하고 추천 서비스와 연동합니다.

주요 구성요소:
    - recommend: 추천 요청을 처리하는 엔드포인트 함수
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from app.services.moment_generator import GeneratorService
from app.schemas.moment_schema import GenerateRequest, GenerateResponse
from app.api.deps import get_moment_generator

router = APIRouter()

@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="게시글 생성",
    description="장소 정보를 기반으로 장소 추천 게시글을 생성합니다."
)
async def generate_moment(
    place_info: GenerateRequest = Body(..., description="오늘의 추천 장소"),
    generator: GeneratorService = Depends(get_moment_generator)
) -> GenerateResponse:
    """
    추천 요청을 처리하는 엔드포인트
    
    이 엔드포인트는 다음과 같은 단계로 동작합니다:
    1. 사용자의 추천 요청을 받음
    2. RecommenderService를 사용하여 추천 생성
    3. 추천 결과를 응답 형식에 맞게 변환하여 반환
    
    Args:
        text (str): 사용자의 추천 요청 키워드
        recommender (RecommenderService): 의존성으로 주입된 추천 서비스
        metrics (RecommendMetrics): 의존성으로 주입된 추천 메트릭스
        
    Returns:
        RecommendResponse: 추천 결과 데이터
        
    Raises:
        HTTPException: 추천 생성 과정에서 오류가 발생한 경우
    """
    try:
        return await generator.generate_moment(place_info=place_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )