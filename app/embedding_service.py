import os
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingService:
    def __init__(self):
        pass

    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        try:
            local_model_path = os.path.join("/app/models", model_name)
            if os.path.exists(local_model_path):
                print("Used local model for embedding")
                model = SentenceTransformer(local_model_path) 
            else:
                raise ReferenceError("Model not found locally")
            return model.encode(texts)
        except Exception as e:
            print(f"Exception: {e}")