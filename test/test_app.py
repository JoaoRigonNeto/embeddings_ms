from fastapi.testclient import TestClient
import os
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_available_models_check():
    with TestClient(app) as client:
        os_available_models = os.listdir("/app/models")
        response = client.get("/available_models")
        assert response.status_code == 200
        assert response.json().get("models") == os_available_models

def test_model_info_check():
    with TestClient(app) as client:
        available_models = client.get("/available_models").json().get("models")
        for model in available_models:
            response = client.get(f"/info/{model}")
            assert response.status_code == 200

            info = response.json()
            assert "info" in info
            assert isinstance(info["info"], dict)

def test_embedding_single_imput_check():
    with TestClient(app) as client:
        available_models = client.get("/available_models").json().get("models")
        for model in available_models:
            model_info = client.get(f"/info/{model}").json().get("info")
            response = client.post("/v1/embeddings", json={
                    "input": "This is a test sentence.",
                    "model": model
            })
            assert response.status_code == 200

            data = response.json()
            assert data["model"] == model
            assert data["object"] == "list"
            assert "data" in data
            assert len(data["data"]) == 1
            assert isinstance(data["data"][0]["embedding"], list)
            assert len(data["data"][0]["embedding"]) ==  model_info.get("dimensions", -1)

def test_embedding_multiple_imput_check():
    with TestClient(app) as client:
        available_models = client.get("/available_models").json().get("models")
        for model in available_models:
            model_info = client.get(f"/info/{model}").json().get("info")
            response = client.post("/v1/embeddings", json={
                    "input": ["This is the first test sentence.", "This is the second test sentence"],
                    "model": model
            })
            assert response.status_code == 200

            data = response.json()
            assert data["model"] == model
            assert data["object"] == "list"
            assert "data" in data
            assert len(data["data"]) == 2
            assert isinstance(data["data"][0]["embedding"], list)
            assert isinstance(data["data"][1]["embedding"], list)
            assert len(data["data"][0]["embedding"]) ==  model_info.get("dimensions", -1)
            assert len(data["data"][1]["embedding"]) ==  model_info.get("dimensions", -1)