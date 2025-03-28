import numpy as np
from typing import List
from model_manager import get_model


class EmbeddingService:
    def __init__(self):
        pass

    def generate_embeddings(self, texts: List[str], model_name:str) -> np.ndarray:
        local_model = get_model(model_name=model_name)
        return local_model.encode(texts)