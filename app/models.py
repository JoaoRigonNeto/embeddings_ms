import os
from sentence_transformers import SentenceTransformer

def download_and_cache_models():
    models = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ]

    download_dir = "/model-download/models"
    os.makedirs(download_dir, exist_ok=True)

    for model_name in models:
        try:
            print(f"Downloading model: {model_name}")
            model_path = os.path.join(download_dir, model_name)
            model = SentenceTransformer(model_name)
            model.save(model_path)
            print(f"Model {model_name} downloaded  and saved successfully")
        except Exception as e:
            print(f"Error : {e}")

if __name__ == "__main__":
    download_and_cache_models()
