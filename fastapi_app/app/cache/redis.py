"""
Redis 캐시 모듈

이 모듈은 Redis를 사용한 캐시 기능을 제공합니다.
Redis 클라이언트를 관리하고 의존성을 제공합니다.

주요 구성요소:
    - get_redis_client: Redis 클라이언트 의존성
"""

from redis import Redis
from fastapi import Depends, HTTPException
from app.core.config import settings

def get_redis_client() -> Redis:
    """
    Redis 클라이언트 의존성
    
    Returns:
        Redis: Redis 클라이언트 인스턴스
        
    Raises:
        HTTPException: Redis 연결 실패 시
    """
    try:
        return Redis.from_url(settings.REDIS_URL)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Redis 연결 실패: {str(e)}"
        )
