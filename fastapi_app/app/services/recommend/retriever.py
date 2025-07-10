"""
장소 벡터 저장소 모듈

이 모듈은 ChromaDB를 사용하여 장소 정보를 벡터 형태로 저장하고 검색하는 기능을 제공합니다.
주요 구성요소:
    - PlaceStore: 장소 벡터 저장소 클래스
"""

import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3

import os
import logging
import chromadb
import numpy as np

from typing import Optional, Dict, List, Any
from fastapi import Depends

from app.core.config import settings
from app.core.constants import CATEGORY_MAP

class PlaceStore:
    """
    장소 벡터 저장소 클래스
    
    이 클래스는 ChromaDB를 사용하여 장소 정보를 벡터 형태로 저장하고 검색합니다.
    
    Attributes:
        client (chromadb.PersistentClient): ChromaDB 클라이언트
        category_map (Dict[str, str]): 카테고리 매핑
    """
    
    def __init__(self, logger=None):  # None으로 지정하는 이유는 추후 의존성 주입의 유연성을 위함 (예: 테스트 환경에서는 로거를 직접 전달할 수 있음)
        """PlaceStore 초기화"""
        if logger is None:
            from app.logging.di import get_logger_dep
            logger = get_logger_dep()
        self.logger = logger
        db_path = settings.VECTOR_STORE_PATH

        # if not os.path.exists(db_path) or not os.listdir(db_path):
        #     from app.data.chroma_db import make_chroma_db
        #     make_chroma_db()

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=db_path)
        self.category_map = CATEGORY_MAP

        # 컬렉션 초기화
        self.collections = {}
        self._init_collections()
    
    def _init_collections(self):
        """
        ChromaDB 컬렉션 초기화
        
        각 카테고리별로 컬렉션을 생성하고, 없는 경우 새로 생성합니다.
        """
        for category in self.category_map.values():
            try:
                # 기존 컬렉션 확인
                collection = self.client.get_collection(name=category)
                self.collections[category] = collection
            except Exception as e:
                raise Exception(f"컬렉션 초기화 실패: {str(e)}")
    
    def search_places(
        self,
        category: str,
        keyword_vec: List[float],
        n_results: Optional[int] = 50
    ) -> Optional[Dict[str, Any]]:
        """
        키워드와 유사한 장소 검색
        
        Args:
            category (str): 검색할 카테고리
            keyword_vec (List[float]): 검색 키워드 벡터
            n_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
            
        Raises:
            ValueError: 유효하지 않은 카테고리인 경우
            Exception: 검색 중 오류 발생 시
        """            
        try:
            collection = self.collections.get(self.category_map[category])
            if collection is None:
                raise ValueError(f"컬렉션 미존재: {category}")
    
            results = collection.query(
                query_embeddings=[keyword_vec],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 검증
            if not results or not results.get('metadatas') or not results['metadatas'][0]:
                self.logger.warning("검색 결과가 없습니다.")
                return None
                
            return results
        except Exception as e:
            self.logger.error(f"장소 검색 중 오류 발생: {str(e)}")
            raise Exception(f"장소 검색 중 오류 발생: {str(e)}")