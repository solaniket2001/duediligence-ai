import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert "engine" in response.json()

def test_analyze_endpoint_validation():
    # Missing 'ticker' which is a required Pydantic field
    bad_payload = {
        "company_name": "Tesla, Inc.",
        "peer_ticker": "MSFT",
    }
    response = client.post("/api/analyze", json=bad_payload)
    
    # FastAPI should automatically reject this with a 422 Unprocessable Entity
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "ticker"]
    assert response.json()["detail"][0]["msg"] == "Field required"