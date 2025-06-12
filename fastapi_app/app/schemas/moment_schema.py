"""
기록 데이터 모델 정의

이 모듈은 기록 관련 서비스에서 사용되는 요청/응답 데이터 구조를 정의합니다.
Pydantic을 사용하여 데이터 검증과 직렬화/역직렬화를 처리합니다.

주요 구성요소:
    - GenerateRequest: 게시글 생성 요청 데이터 모델
    - GenerateResponse: 게시글 생성 응답 데이터 모델
"""

from pydantic import BaseModel
from typing import Optional, List


class OpeningHour(BaseModel):
    """
    요일별 운영 시간 정보

    Attributes:
        day (str): 요일 (예: 'mon', 'tue')
        hours (Optional[str]): 운영 시간 (예: '08:00~17:00')
        breakTime (Optional[str]): 미운영 시간 (예: '08:00~17:00')
    """
    day: str
    hours: Optional[str] = None
    breakTime: Optional[str] = None


class OpeningHours(BaseModel):
    """
    전체 운영 시간 정보

    Attributes:
        status (str): 현재 영업 상태 (예: '영업 중')
        schedules (List[OpeningHour]): 요일별 운영 시간 리스트
    """
    status: str
    schedules: List[OpeningHour]


class MenuItem(BaseModel):
    """
    메뉴 항목 정보

    Attributes:
        name (str): 메뉴 이름
        price (Optional[int]): 메뉴 가격 (단위: 원)
    """
    name: str
    price: Optional[int] = None


class GenerateRequest(BaseModel):
    """
    게시글 생성 요청 데이터 모델

    Attributes:
        id (int): 장소 식별자
        name (str): 장소 이름
        address (Optional[str]): 장소 도로명주소
        keyword (List[str]): 키워드 리스트 (예: '포장', '주차')
        description (Optional[str]): 장소 설명
        opening_hours (OpeningHours): 운영 시간 정보
        phone (Optional[str]): 전화번호
        menu (List[MenuItem]): 메뉴 리스트
    """
    id: int
    name: str
    address: Optional[str] = None
    thumbnail: Optional[str] = None
    keyword: List[str]
    description: Optional[str] = None
    opening_hours: OpeningHours
    phone: Optional[str] = None
    menu: List[MenuItem]


class GenerateResponse(BaseModel):
    """
    게시글 생성 응답 데이터 모델

    Attributes:
        title (str): 생성된 게시글 제목
        content (str): 생성된 게시글 본문
        place_id (str): 대상 장소 ID
        images (List[str]): 이미지 URL 리스트
        is_public (bool): 게시글 공개 여부 (기본값: True)
    """
    title: str
    content: str
    place_id: int
    images: List[str] = []
    is_public: bool = True
