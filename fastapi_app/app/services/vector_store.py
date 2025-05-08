"""
벡터 저장소 서비스 모듈

이 모듈은 Chroma DB를 사용하여 벡터 저장소를 관리합니다.
문서의 임베딩을 생성하고 저장하며, 유사도 검색을 수행합니다.

주요 구성요소:
    - VectorStore: 벡터 저장소 관리 클래스
"""

import os

# chroma 사용 시 python-sqlite3 버전 문제 해결
# 내장 모듈 sqlite3 대신 pysqlite3 사용
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import sqlite3

from typing import List, Dict, Any, Optional
from chromadb import Client, Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

from app.core.config import settings

class VectorStore:
    """
    벡터 저장소 관리 클래스
    
    Attributes:
        client (Client): Chroma DB 클라이언트
        collection: Chroma DB 컬렉션
        embedding_model: 임베딩 모델
    """
    
    def __init__(self):
        """벡터 저장소 초기화"""
        # Chroma DB 클라이언트 초기화
        self.client = Client(Settings(
            persist_directory=settings.VECTOR_STORE_PATH,
            is_persistent=True
        ))
        
        # 임베딩 모델 초기화
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_function = None
        
        # 컬렉션 생성 또는 로드
        self.collection = self.client.get_or_create_collection(
            name=settings.VECTOR_STORE_COLLECTION_NAME,
            embedding_function=self._get_embedding_function()
        )
    
    @property
    def embedding_model(self) -> SentenceTransformer:
        """임베딩 모델 프로퍼티"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        return self._embedding_model
    
    def _get_embedding_function(self):
        """임베딩 함수 반환"""
        if self._embedding_function is None:
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL_NAME
            )
        return self._embedding_function
    
    def add_documents(self, documents: List[Dict[str, Any]], collection_name: str = None) -> None:
        """
        문서 추가 (컬렉션 지정 가능)
        Args:
            documents (List[Dict[str, Any]]): 추가할 문서 리스트
            collection_name (str): 추가할 컬렉션명 (기본값: 설정값)
        """
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        col = self.collection if collection_name is None else self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._get_embedding_function()
        )
        col.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(self, query: str, n_results: int = 5, collection_name: str = None) -> List[Dict[str, Any]]:
        """
        유사도 검색 수행 (컬렉션 지정 가능)
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            collection_name (str): 검색할 컬렉션명 (기본값: 설정값)
        Returns:
            List[Dict[str, Any]]: 검색 결과 리스트
        """
        col = self.collection if collection_name is None else self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._get_embedding_function()
        )
        results = col.query(
            query_texts=[query],
            n_results=n_results
        )
        return [
            {
                "id": id,
                "text": text,
                "metadata": metadata,
                "distance": distance
            }
            for id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        문서 삭제
        
        Args:
            ids (List[str]): 삭제할 문서 ID 리스트
        """
        self.collection.delete(ids=ids)
    
    def get_document(self, id: str) -> Dict[str, Any]:
        """
        단일 문서 조회
        
        Args:
            id (str): 문서 ID
            
        Returns:
            Dict[str, Any]: 문서 정보
        """
        result = self.collection.get(ids=[id])
        if not result["ids"]:
            return None
            
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0]
        }
    
    def close(self) -> None:
        """리소스 정리"""
        if self._embedding_model is not None:
            # 임베딩 모델의 리소스 정리
            del self._embedding_model
            self._embedding_model = None
        # Chroma DB 클라이언트 정리
        if hasattr(self, 'client'):
            del self.client
