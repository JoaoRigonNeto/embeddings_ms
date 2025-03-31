import numpy as np
from typing import List
from model_manager import get_model, get_available_models


class EmbeddingService:
    def __init__(self):
        pass


    def is_model_available(self, model_name: str) -> bool:
        return model_name in get_available_models


    def generate_embeddings(self, texts: List[str], model_name:str) -> np.ndarray:
        local_model = get_model(model_name=model_name)
        return local_model.encode(texts)