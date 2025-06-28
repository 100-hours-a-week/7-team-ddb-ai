import threading

from app.core.config import settings
from app.services.recommend.embedding import EmbeddingModel

class EmbeddingModelFactory:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> EmbeddingModel:
        """
        임베딩 모델의 싱글톤 인스턴스를 반환합니다.
        
        Returns:
            EmbeddingModel: 임베딩 모델 인스턴스
            
        Raises:
            RuntimeError: 임베딩 모델 초기화 실패 시
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = EmbeddingModel(
                            settings.ONNX_MODEL_PATH,
                            settings.TOKENIZER_PATH
                        )
                    except Exception as e:
                        raise RuntimeError(f"임베딩 모델 초기화 실패: {str(e)}")
        return cls._instance 