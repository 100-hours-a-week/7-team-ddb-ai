import threading
from app.services.vector_store import PlaceStore

class PlaceStoreFactory:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> PlaceStore:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = PlaceStore()
                    except Exception as e:
                        raise RuntimeError(f"PlaceStore 초기화 실패: {str(e)}")
        return cls._instance
