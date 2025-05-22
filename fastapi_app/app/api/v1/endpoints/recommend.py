"""
추천 API 엔드포인트

이 모듈은 추천 서비스의 REST API 엔드포인트를 제공합니다.
FastAPI를 사용하여 HTTP 요청을 처리하고 추천 서비스와 연동합니다.

주요 구성요소:
    - recommend: 추천 요청을 처리하는 엔드포인트 함수
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.services.recommender import RecommenderService
from app.schemas.recommend_schema import RecommendResponse
from app.api.deps import get_recommender, get_recommend_metrics  # 추천 API 메트릭 의존성 함수 임포트

router = APIRouter()

@router.get(
    "",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="장소 추천",
    description="사용자 입력을 기반으로 장소를 추천합니다."
)
async def get_recommendation(
    text: str = Query(..., description="추천을 위한 키워드나 문장"),
    recommender: RecommenderService = Depends(get_recommender),
    metrics = Depends(get_recommend_metrics)  # 메트릭 객체를 의존성 주입으로 받음
) -> RecommendResponse:
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
        recommender.metrics = metrics  # 엔드포인트에서 RecommenderService에 메트릭 객체 주입
        return await recommender.get_recommendation(user_input=text)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )