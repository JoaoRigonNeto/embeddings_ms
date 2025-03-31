from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_embedding_single_imput():
    response = client.post("/v1/embeddings", json={
        "input": "This is a test sentence.",
        "model": "all-MiniLM-L6-v2"
    })
    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "all-MiniLM-L6-v2"
    assert data["object"] == "list"
    assert "data" in data
    assert len(data["data"]) == 1
    assert isinstance(data["data"][0]["embedding"], list)
    assert len(data["data"][0]["embedding"]) == 384