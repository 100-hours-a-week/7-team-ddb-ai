"""
로깅 설정 모듈

이 모듈은 애플리케이션의 로깅 설정을 관리합니다.
Python의 logging 모듈을 사용하여 로깅을 구성합니다.

주요 구성요소:
    - setup_logger: 로거 설정 함수
    - get_logger: 로깅 의존성
"""

import logging
from fastapi import Depends
from app.core.config import settings

def setup_logger() -> logging.Logger:
    """
    로거 설정
    
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    logger = logging.getLogger("app")
    logger.setLevel(settings.LOG_LEVEL)
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(settings.LOG_LEVEL)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(console_handler)
    
    return logger

def get_logger() -> logging.Logger:
    """
    로깅 의존성
    
    Returns:
        logging.Logger: 설정된 로거 인스턴스
    """
    return setup_logger()
