"""
API 라우터 모듈

이 모듈은 API 버전 1의 라우터를 정의합니다.
각 도메인별 라우터를 통합하여 API 구조를 관리합니다.
"""

from fastapi import APIRouter
from app.api.v1.endpoints.recommend import router as recommend_router
from app.api.v1.endpoints.moment.generate import router as moment_generate_router
from app.api.v1.endpoints.data.upload import router as data_upload_router

# API v1 라우터 생성
router = APIRouter(prefix="/v1")

# 도메인별 라우터 등록
router.include_router(
    recommend_router,
    prefix="/recommend",  # 도메인 중심의 명확한 경로
    tags=["recommend"],
    responses={
        200: {"description": "성공"},
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 내부 오류"}
    }
)

router.include_router(
    moment_generate_router,
    prefix="/moment",
    tags=["moment"],
    responses={
        200: {"description": "성공"},
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 내부 오류"}
    }
)

router.include_router(
    data_upload_router,
    prefix="/data",
    tags=["data"],
    responses={
        200: {"description": "성공"},
        400: {"description": "잘못된 요청"},
        500: {"description": "서버 내부 오류"}
    }
)