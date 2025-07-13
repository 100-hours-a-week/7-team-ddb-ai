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
from huggingface_hub import login

# .env 파일 로드
load_dotenv()

class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스
    
    Attributes:
        MODEL_NAME (str): 사용할 LLM 모델 이름
        GOOGLE_API_KEY (str): Google API 키
        TEMPERATURE (float): LLM temperature 값
        VECTOR_STORE_PATH (str): 벡터 저장소 경로
        EMBEDDING_MODEL_NAME (str): 임베딩 모델 이름
        VECTOR_STORE_COLLECTION_NAME (str): 벡터 저장소 컬렉션 이름
        
    TODO: 추후 구현 예정
        REDIS_URL (str): Redis 연결 URL
        LOG_LEVEL (str): 로깅 레벨
    """

    # CLOVAX 설정
    CLOVAX_MODEL_NAME: str = os.getenv("CLOVAX_MODEL_NAME", "chanhue/dolpin-hyperclova-lora")
    
    # LLM 설정
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-2.0-flash-lite")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    TEMPERATURE: float = os.getenv("TEMPERATURE", 0.7)
    
    # 벡터 저장소 설정
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    VECTOR_STORE_COLLECTION_NAME: str = os.getenv("VECTOR_STORE_COLLECTION_NAME", "documents")

    # ONNX 임베딩 모델 설정
    ONNX_MODEL_PATH: str = os.getenv("ONNX_MODEL_PATH", "app/model/snunlp_KR-SBERT-V40K-klueNLI-augSTS_quant.onnx")
    TOKENIZER_PATH: str = os.getenv("TOKENIZER_PATH", "app/model/krsbert_tokenizer")

    # 카카오 API 설정
    KAKAO_API_KEY: str = os.getenv("KAKAO_API_KEY")

    # S3 권한 설정
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY")
    S3_DEFAULT_REGION: str = os.getenv("S3_DEFAULT_REGION")
    
    # 장소 이미지 저장 경로 설정
    TEMP_IMAGE_PATH: str = os.getenv("TEMP_IMAGE_PATH", "app/data/temp_image")
    S3_IMAGE_PATH: str = os.getenv("S3_IMAGE_PATH", "s3-dolpin-image-dev/place")
    S3_METADATA_PATH: str = os.getenv("S3_METADATA_PATH", "ai-metadata-temp-bucket")

    # 장소 데이터 API 요청 시크릿 키 설정
    UPLOAD_SECRET_KEY: str = os.getenv("UPLOAD_SECRET_KEY")

    # TODO: 추후 구현 예정
    # # Redis 설정
    # REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # # 로깅 설정
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 필수 설정 값 검증
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY가 설정되지 않았습니다.")
        if not self.KAKAO_API_KEY:
            raise ValueError("KAKAO_API_KEY 설정되지 않았습니다.")
        if not self.S3_ACCESS_KEY:
            raise ValueError("S3_ACCESS_KEY 설정되지 않았습니다.")
        if not self.S3_SECRET_KEY:
            raise ValueError("S3_SECRET_KEY 설정되지 않았습니다.")
        if not self.S3_DEFAULT_REGION:
            raise ValueError("S3_DEFAULT_REGION 설정되지 않았습니다.")
        if not self.UPLOAD_SECRET_KEY:
            raise ValueError("UPLOAD_SECRET_KEY 설정되지 않았습니다.")
        
        # 벡터 저장소 디렉토리 생성
        os.makedirs(self.VECTOR_STORE_PATH, exist_ok=True)

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# 전역 설정 인스턴스 생성
settings = Settings()
