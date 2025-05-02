"""
추천 API 엔드포인트

이 모듈은 추천 서비스의 REST API 엔드포인트를 제공합니다.
FastAPI를 사용하여 HTTP 요청을 처리하고 추천 서비스와 연동합니다.

주요 구성요소:
    - recommend: 추천 요청을 처리하는 엔드포인트 함수
"""

from fastapi import APIRouter, Depends, HTTPException
from app.services.recommender import RecommenderService
from app.schemas.recommend_schema import RecommendRequest, RecommendResponse, Recommendation
from app.api.deps import get_recommender

router = APIRouter()

@router.post("/recommend", response_model=RecommendResponse)
async def recommend(
    request: RecommendRequest,
    recommender: RecommenderService = Depends(get_recommender)
):
    """
    추천 요청을 처리하는 엔드포인트
    
    이 엔드포인트는 다음과 같은 단계로 동작합니다:
    1. 사용자의 추천 요청을 받음
    2. RecommenderService를 사용하여 추천 생성
    3. 추천 결과를 응답 형식에 맞게 변환하여 반환
    
    Args:
        request (RecommendRequest): 사용자의 추천 요청 데이터
        recommender (RecommenderService): 의존성으로 주입된 추천 서비스
        
    Returns:
        RecommendResponse: 추천 결과 데이터
        
    Raises:
        HTTPException: 추천 생성 과정에서 오류가 발생한 경우
    """
    try:
        recommendations = await recommender.get_recommendation(
            user_input=request.text
        )
        
        # 추천 목록을 Recommendation 객체 리스트로 변환
        recommendation_objects = [
            Recommendation(**rec) for rec in recommendations
        ]
        
        return RecommendResponse(recommendations=recommendation_objects)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
