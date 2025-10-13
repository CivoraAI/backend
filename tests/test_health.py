from fastapi.testclient import TestClient
from src.api.app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"
