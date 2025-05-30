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

from typing import Dict, List, Any
import os
import logging
import chromadb
import numpy as np

from typing import Optional
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.constants import CATEGORY_MAP
from app.data.chroma_db import make_chroma_db
from fastapi import Depends
# from app.api.deps import get_logger_dep  # 삭제
# from app.logging.config import get_logger  # 이미 삭제됨

class PlaceStore:
    """
    장소 벡터 저장소 클래스
    
    이 클래스는 ChromaDB를 사용하여 장소 정보를 벡터 형태로 저장하고 검색합니다.
    
    Attributes:
        client (chromadb.PersistentClient): ChromaDB 클라이언트
        embedding_model (SentenceTransformer): 문장 임베딩 모델
        category_map (Dict[str, str]): 카테고리 매핑
    """
    
    def __init__(self, logger=None):  # None으로 지정하는 이유는 추후 의존성 주입의 유연성을 위함 (예: 테스트 환경에서는 로거를 직접 전달할 수 있음)
        """PlaceStore 초기화"""
        if logger is None:
            from app.logging.di import get_logger_dep
            logger = get_logger_dep()
        self.logger = logger
        db_path = settings.VECTOR_STORE_PATH

        if not os.path.exists(db_path) or not os.listdir(db_path):
            make_chroma_db()

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        self.category_map = CATEGORY_MAP

        # 컬렉션 초기화
        self._init_collections()
    
    def _init_collections(self):
        """
        ChromaDB 컬렉션 초기화
        
        각 카테고리별로 컬렉션을 생성하고, 없는 경우 새로 생성합니다.
        """
        try:
            for category in self.category_map.values():
                try:
                    # 기존 컬렉션 확인
                    self.client.get_collection(name=category)
                except Exception as e:
                    raise Exception(f"컬렉션 미존재: {str(e)}")
        except Exception as e:
            raise Exception(f"컬렉션 초기화 실패: {str(e)}")
    
    def encode_text(self, text: str) -> List[float]:
        """
        텍스트를 벡터로 인코딩
        
        Args:
            text (str): 인코딩할 텍스트
            
        Returns:
            List[float]: 인코딩된 벡터
        """
        return self.embedding_model.encode(text).tolist()
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """
        두 벡터 간의 코사인 유사도 계산
        
        Args:
            a (List[float]): 첫 번째 벡터
            b (List[float]): 두 번째 벡터
            
        Returns:
            float: 코사인 유사도
        """
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search_places(
        self,
        category: str,
        keyword: str,
        n_results: Optional[int] = 50
    ) -> Dict[str, Any]:
        """
        키워드와 유사한 장소 검색
        
        Args:
            category (str): 검색할 카테고리
            keyword (str): 검색 키워드
            n_results (int): 반환할 결과 수
            
        Returns:
            Dict[str, Any]: 검색 결과
            
        Raises:
            ValueError: 유효하지 않은 카테고리인 경우
            Exception: 검색 중 오류 발생 시
        """            
        try:
            # logger.info(f"장소 검색 시작: 카테고리={category}, 키워드={keyword}")
            collection_name = self.category_map[category]
            
            try:
                collection = self.client.get_collection(name=collection_name)
                # logger.info(f"컬렉션 '{collection_name}' 로드 완료")
            except Exception as e:
                self.logger.error(f"컬렉션 '{collection_name}' 로드 실패: {str(e)}")
                raise

            try:
                keyword_vec = self.encode_text(keyword)
                # logger.info("키워드 임베딩 완료")
            except Exception as e:
                self.logger.error(f"키워드 임베딩 실패: {str(e)}")
                raise
            
            try:
                # logger.info(f"벡터 검색 시작: n_results={n_results}")
                results = collection.query(
                    query_embeddings=[keyword_vec],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                # logger.info(f"벡터 검색 완료: {len(results['metadatas'][0])}개 결과")
                
                # 결과 검증
                if not results or not results.get('metadatas') or not results['metadatas'][0]:
                    self.logger.warning("검색 결과가 없습니다.")
                    return {
                        'metadatas': [[]],
                        'distances': [[]],
                        'documents': [[]]
                    }
                
                return results, keyword_vec
                
            except Exception as e:
                self.logger.error(f"벡터 검색 실패: {str(e)}")
                raise
            
        except KeyError:
            self.logger.error(f"유효하지 않은 카테고리: {category}")
            raise ValueError(f"유효하지 않은 카테고리: {category}")
        except Exception as e:
            self.logger.error(f"장소 검색 중 오류 발생: {str(e)}")
            raise Exception(f"장소 검색 중 오류 발생: {str(e)}")
    
    def close(self) -> None:
        """
        벡터 저장소 연결 종료
        """
        try:
            # ChromaDB는 자동으로 변경사항을 저장하므로 별도의 persist 호출이 필요 없음
            pass
        except Exception as e:
            raise Exception(f"벡터 저장소 종료 실패: {str(e)}") 