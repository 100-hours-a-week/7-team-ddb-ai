from app.logging.config import get_logger
import logging

def get_logger_dep() -> logging.Logger:
    """
    FastAPI 의존성 주입용 로거 반환 함수
    """
    return get_logger() 