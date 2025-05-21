"""
추천 서비스의 데이터 모델 정의

이 모듈은 추천 서비스에서 사용되는 요청/응답 데이터 구조를 정의합니다.
Pydantic을 사용하여 데이터 검증과 직렬화/역직렬화를 처리합니다.

주요 구성요소:
    - RecommendRequest: 추천 요청 데이터 모델
    - Recommendation: 개별 추천 항목 데이터 모델
    - RecommendResponse: 추천 응답 데이터 모델
"""

from pydantic import BaseModel
from typing import Optional, List

class RecommendRequest(BaseModel):
    """
    추천 요청 데이터 모델
    
    Attributes:
        text (str): 사용자의 추천 요청 텍스트
    """
    text: str

class Recommendation(BaseModel):
    """
    개별 추천 항목 데이터 모델
    
    Attributes:
        id (int): 추천 장소의 고유 식별자
        similarity_score (float): 추천 장소와 사용자 요청 간의 유사도 점수
        keyword (List[int]): 해당 추천 장소에 관련된 키워드 리스트
    """
    id: int
    similarity_score: float
    keyword: List[str]

class RecommendResponse(BaseModel):
    """
    추천 응답 데이터 모델
    
    Attributes:
        recommendations (List[Recommendation]): 추천 장소 목록
    """
    recommendations: List[Recommendation]


