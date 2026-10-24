from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Welcome" in data["message"]
    assert "health" in data


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "storage_backend" in data


def test_search_endpoint_contract():
    payload = {
        "query": "Object oriented programming",
        "top_k": 3,
        "mode": "bm25_only",
        "rerank": False,
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Object oriented programming"
    assert "hits" in data
    assert "execution_time_ms" in data
