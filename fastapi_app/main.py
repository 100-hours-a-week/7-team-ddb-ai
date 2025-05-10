import logging
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints.recommend import router as recommend_router

app = FastAPI(title="추천 서비스 API")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(recommend_router, prefix="/api/v1", tags=["recommendations"])

@app.get("/")
async def root():
    return {"message": "추천 서비스 API가 실행 중입니다."}

@app.get("/health")
async def health_check():
    logger.info("헬스체크 엔드포인트 호출됨")
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }