import threading
from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingModelFactory:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> SentenceTransformer:
        """
        임베딩 모델의 싱글톤 인스턴스를 반환합니다.
        
        Returns:
            SentenceTransformer: 임베딩 모델 인스턴스
            
        Raises:
            RuntimeError: 임베딩 모델 초기화 실패 시
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = SentenceTransformer(
                            settings.EMBEDDING_MODEL_NAME,
                            device='cpu'
                        )
                    except Exception as e:
                        raise RuntimeError(f"임베딩 모델 초기화 실패: {str(e)}")
        return cls._instance 