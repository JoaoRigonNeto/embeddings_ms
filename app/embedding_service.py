import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from .model_manager import get_model

class EmbeddingService:
    def __init__(self):
        pass

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            local_model = get_model()
            return local_model.encode(texts)
        except Exception as e:
            print(f"Exception: {e}")