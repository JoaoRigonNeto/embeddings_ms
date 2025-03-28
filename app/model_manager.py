import os
import json
import logging
import argparse
import requests
from sentence_transformers import SentenceTransformer
from huggingface_hub import configure_http_backend
import urllib3

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)

models = {}
info = {}


def load_info() -> None:
    global info
    with open('info.json') as json_data:
        local_info = json.load(json_data)
        models_dir = "/app/models"
        for model_name in os.listdir(models_dir):
            logging.info(f"Loading model '{model_name}' infos.")
            info[model_name] = local_info.get(model_name, {})


def get_info(model_name: str) -> dict[str, str]:
    global info
    if model_name not in info:
        raise RuntimeError(f"Model '{model_name}' info not loaded.")
    return info[model_name]


def load_models() -> None:
    global models
    models_dir = "/app/models"
    for local_model in os.listdir(models_dir):
        if local_model not in models:
            models[local_model] = SentenceTransformer(os.path.join(models_dir, local_model))


def get_model(model_name: str) -> SentenceTransformer:
    global models
    if model_name not in models:
        raise RuntimeError(f"Model '{model_name}' not loaded.")
    return models[model_name]

def _download_and_cache_model(models_dir: str, model_name: str ="all"):

    if model_name == "all":
        model_name = [
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "distiluse-base-multilingual-cased-v2",
            "xlm-r-distilroberta-base-paraphrase-v1",
            "paraphrase-xlm-r-multilingual-v1",
            "all-distilroberta-v1",
            "all-MiniLM-L12-v2",
            "all-mpnet-base-v2",
            "msmarco-distilbert-base-tas-b",
        ]
    else:
        model_name = [model_name]

    os.makedirs(models_dir, exist_ok=True)
    try:
        for model in model_name:
            logging.info(f"Downloading model: {model}")
            model_path = os.path.join(models_dir, model)
            local_model = SentenceTransformer(model)
            local_model.save(model_path)
            logging.info(f"Model {model} downloaded  and saved successfully")
    except Exception as e:
        logging.error(f"Error : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("model_manager")
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()
    _download_and_cache_model(models_dir="/model-download/models", model_name=args.model_name)
