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
    with open('models_info.json') as json_data:
        local_info = json.load(json_data)
        models_dir = "/app/models"
        for model_name in os.listdir(models_dir):
            logging.info(f"Loading model '{model_name}' infos.")
            info[model_name] = local_info.get(model_name, {})


def get_info(model_name: str) -> dict[str, str]:
    global info
    if model_name not in info:
        raise RuntimeError(f"Model '{model_name}' information not loaded.")
    return info[model_name]


def load_models() -> None:
    global models
    models_dir = "/app/models"
    try:
        local_models = os.listdir(models_dir)
    except FileNotFoundError:
        local_models = []

    if not local_models:
        _download_and_cache_model(models_dir="/app/models", model_name=os.environ.get("MODEL_NAME", "none")) 
        local_models = os.listdir(models_dir)

    for local_model in local_models:
        if local_model not in models:
            models[local_model] = SentenceTransformer(os.path.join(models_dir, local_model), trust_remote_code=True)
            logging.info(f"Model '{local_model}' loaded into memory")
        else:
            logging.info(f"Model '{local_model}' already loaded into memory")


def get_model(model_name: str) -> SentenceTransformer:
    global models
    if model_name not in models:
        raise RuntimeError(f"Model '{model_name}' not available.")
    return models[model_name]


def get_available_models() -> list[str]:
    global models
    return list(models.keys())


def _download_and_cache_model(models_dir: str, model_name: str):

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
            "Alibaba-NLP/gte-multilingual-base"
        ]
    elif not model_name or model_name == "none":
        logging.info("Skipping model download...")
        return
    else:
        model_name = [model_name]

    os.makedirs(models_dir, exist_ok=True)
    for model in model_name:
        try:
            logging.info(f"Downloading model: {model}")
            model_path = os.path.join(models_dir, model.split("/")[-1])
            local_model = SentenceTransformer(model, trust_remote_code=True)
            local_model.save(model_path)
            logging.info(f"Model {model} downloaded  and saved successfully")
        except Exception as e:
            logging.error(f"Failed to download model {model}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("model_manager")
    parser.add_argument("--model-name", type=str)
    args = parser.parse_args()
    _download_and_cache_model(models_dir="/model-download/models", model_name=args.model_name)
