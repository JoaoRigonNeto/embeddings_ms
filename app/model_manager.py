import os
import logging
from sentence_transformers import SentenceTransformer

model = None
logging.basicConfig(level=logging.INFO)

def load_model():
    global model
    models_dir = "/app/models"
    model_name = os.environ.get("MODEL_NAME", False)

    if not model_name:
        raise ValueError("Environment variable MODEL_NAME must be set")

    if not os.path.isdir(os.path.join(models_dir, model_name)):
        _download_and_cache_model(models_dir, model_name)

    if model is None:
        model = SentenceTransformer(os.path.join(models_dir, model_name))
        logging.info("Loaded SentenceTransformer model")

def get_model():
    global model
    if model is None:
        raise RuntimeError("Model must be loaded first")
    return model

def _download_and_cache_model(models_dir: str, model_name: str):

    os.makedirs(models_dir, exist_ok=True)
    try:
        logging.info(f"Downloading model: {model_name}")
        model_path = os.path.join(models_dir, model_name)
        local_model = SentenceTransformer(model_name)
        local_model.save(model_path)
        logging.info(f"Model {model_name} downloaded  and saved successfully")
    except Exception as e:
        logging.error(f"Error : {e}")
