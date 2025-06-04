import threading
from sentence_transformers import SentenceTransformer
from app.core.config import settings

_embedding_model = None
_embedding_model_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device='cpu')
    return _embedding_model 