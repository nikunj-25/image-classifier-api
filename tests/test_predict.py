from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    with open("tests/sample.jpg", "rb") as image_file:
        response = client.post("/predict", files={"file": image_file})
    assert response.status_code == 200
    assert "label" in response.json()