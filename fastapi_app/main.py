import time
import logging
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import router as api_v1_router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="장소 추천 API",
    description="키워드 기반 장소 추천 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# API 라우터 등록
app.include_router(api_v1_router, prefix="/api")

# 요청/응답 로깅 미들웨어
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    try:
        logger.info(f"요청 시작: {request.method} {request.url}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"요청 완료: {request.method} {request.url} - 상태: {response.status_code} - 소요시간: {process_time:.2f}초")
        return response
    except Exception as e:
        logger.error(f"요청 처리 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"서버 내부 오류: {str(e)}"}
        )

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