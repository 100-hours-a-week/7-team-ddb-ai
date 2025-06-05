from app.services.embedding_factory import EmbeddingModelFactory

def get_embedding_model():
    """
    임베딩 모델의 싱글톤 인스턴스를 반환합니다.
    
    Returns:
        SentenceTransformer: 임베딩 모델 인스턴스
    """
    return EmbeddingModelFactory.get_instance() 