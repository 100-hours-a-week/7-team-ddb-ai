import numpy as np
import onnxruntime as ort

from typing import List, Union
from transformers import AutoTokenizer

class EmbeddingModel:
    def __init__(self, onnx_model_path: str, tokenizer_path: str, max_length: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_names = {i.name: i.name for i in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name
        self.max_length = max_length

    def _preprocess(self, sentences: Union[str, List[str]]) -> dict:
        if isinstance(sentences, str):
            sentences = [sentences]

        encoded = self.tokenizer(
            sentences,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        return {
            self.input_names["input_ids"]: encoded["input_ids"].astype(np.int64),
            self.input_names["attention_mask"]: encoded["attention_mask"].astype(np.int64)
        }

    def _postprocess(self, outputs: np.ndarray) -> np.ndarray:
        # Mean pooling
        return outputs.mean(axis=1)

    def get_sentence_embedding_dimension(self) -> int:
        output_shape = self.session.get_outputs()[0].shape
        if len(output_shape) == 3:
            return output_shape[2]
        elif len(output_shape) == 2:
            return output_shape[1]
        raise ValueError(f"Unexpected ONNX output shape: {output_shape}")

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        if isinstance(sentences, str):
            input_feed = self._preprocess(sentences)
            output = self.session.run([self.output_name], input_feed)[0]
            return self._postprocess(output)[0]  # Return single vector

        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            input_feed = self._preprocess(batch)
            output = self.session.run([self.output_name], input_feed)[0]
            pooled = self._postprocess(output)
            embeddings.append(pooled)
        return np.vstack(embeddings).tolist()