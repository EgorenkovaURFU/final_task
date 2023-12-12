# TF2 version
#import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel, HttpUrl
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.efficientnet import decode_predictions


class Item(BaseModel):
    url: HttpUrl


app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Welcome!'}

def load_model():
    model = hub.KerasLayer('https://www.kaggle.com/models/google/aiy/frameworks/TensorFlow1/variations/vision-classifier-birds-v1/versions/1')
    return model

def preprocess_image(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.post("/prediction/")
async def get_net_image_prediction(item: Item):
    if item.url == "":
        print(item.url)
        return {"message": "No image link provided"}
    
    img = image.load_img(get_file('image', str(item.url)),target_size=(224, 224))
  

    x = preprocess_image(img)
    

    model = load_model()
    pred = model.predict(x)
    classes = decode_predictions(pred, top=1)[0]
    for i in classes:
        model_prediction = str(i[1])
        model_prediction_confidence_score = str(i[2])

        return {"model-prediction": model_prediction, "model-prediction-confidence-score": model_prediction_confidence_score}