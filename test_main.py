from fastapi.testclient import TestClient
from main import app, load_model, save_image, preprocess_image_to_tensor
from tensorflow import keras

client = TestClient(app)

def test_read_main():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Welcome!'}


def test_predict():
    response = client.post('/prediction/',
                           json={"url": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7fLoE1olbz79Y0DBPkLvm1k8F_RpYfJK0hIXU2QZL_Q&s"})
    json_data = response.json()
    assert response.status_code == 200
    assert json_data['prediction'] == 'Scarlet Tanager'


def test_load_model():
    model = load_model()
    assert str(type(model)) == "<class 'keras.src.engine.sequential.Sequential'>"

