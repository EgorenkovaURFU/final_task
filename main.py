# TF2 version
#import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel, HttpUrl
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


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

