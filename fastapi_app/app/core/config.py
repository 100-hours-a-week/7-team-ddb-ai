"""
애플리케이션 설정 모듈

이 모듈은 애플리케이션의 설정을 관리합니다.
환경 변수를 로드하고 설정 값을 제공합니다.

주요 구성요소:
    - Settings: 애플리케이션 설정 클래스
    - settings: 전역 설정 인스턴스
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    
    Attributes:
        MODEL_NAME (str): 사용할 LLM 모델 이름
        GOOGLE_API_KEY (str): Google API 키
        TEMPERATURE (float): LLM temperature 값
        
    TODO: 추후 구현 예정
        REDIS_URL (str): Redis 연결 URL
        VECTOR_STORE_PATH (str): 벡터 저장소 경로
        LOG_LEVEL (str): 로깅 레벨
    """
    
    # LLM 설정
    MODEL_NAME: str = "gemini-1.5-flash"
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    TEMPERATURE: float = 0.7
    
    # TODO: 추후 구현 예정
    # # Redis 설정
    # REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # # 벡터 저장소 설정
    # VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    
    # # 로깅 설정
    # LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 필수 설정 값 검증
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")

# 전역 설정 인스턴스 생성
settings = Settings()
