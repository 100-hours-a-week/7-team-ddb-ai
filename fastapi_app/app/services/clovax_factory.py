import threading
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from app.core.config import settings

class ClovaXFactory:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def _create_instance(cls):
        model_id = settings.CLOVAX_MODEL_NAME  # 예: "chanhue/dolpin-hyperclova-lora"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256
        )
        return HuggingFacePipeline(pipeline=hf_pipe)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls._create_instance()
        return cls._instance 