import os
import logging
from sentence_transformers import SentenceTransformer

def download_and_cache_models():

    models = [
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

    # non_sentance_transformers_models = [
    #     "neuralmind/bert-base-portuguese-cased",
    #     "pierreguillou/gpt2-small-portuguese",
    # ]

    download_dir = "/model-download/models"
    os.makedirs(download_dir, exist_ok=True)

    for model_name in models:
        try:
            logging.info(f"Downloading model: {model_name}")
            model_path = os.path.join(download_dir, model_name)
            model = SentenceTransformer(model_name)
            model.save(model_path)
            logging.info(f"Model {model_name} downloaded  and saved successfully")
        except Exception as e:
            logging.error(f"Error : {e}")

if __name__ == "__main__":
    download_and_cache_models()
