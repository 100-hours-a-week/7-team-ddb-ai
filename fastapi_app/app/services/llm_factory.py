import threading
from app.core.config import settings

class LLMFactory:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def _create_instance(cls):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE
        )
            
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_instance()
        return cls._instance

        