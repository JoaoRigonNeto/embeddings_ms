import os
import logging
import argparse
from sentence_transformers import SentenceTransformer
import requests
from huggingface_hub import configure_http_backend
import urllib3

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

model = None
logging.basicConfig(level=logging.INFO)

def load_model():
    global model
    models_dir = "/app/models"
    #TODO load all available models (1 or many)
    model_name = "all-MiniLM-L6-v2"

    if model is None:
        model = SentenceTransformer(os.path.join(models_dir, model_name))
        logging.info("Loaded SentenceTransformer model")

def get_model():
    global model
    #TODO Get if many, specific model
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("model_manager")
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()
    _download_and_cache_model(models_dir="/model-download/models", model_name=args.model_name)
