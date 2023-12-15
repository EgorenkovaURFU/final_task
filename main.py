import tensorflow_hub as hub
import tensorflow as tf
import keras

from fastapi import FastAPI

from pydantic import BaseModel, HttpUrl
from tensorflow.keras.preprocessing import image

import requests  
import pandas as pd
import numpy as np


IMAGE_RES = 224 

labels = pd.read_csv("labels_oiseaux.csv", sep=';', header=0, index_col=0)


class Item(BaseModel):
    url: HttpUrl


app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Welcome!'}


def load_model():
    URL = 'https://www.kaggle.com/models/google/aiy/frameworks/TensorFlow1/variations/vision-classifier-birds-v1/versions/1'
    bird = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    bird.trainable=False
    model=tf.keras.Sequential([bird])
    return model


def preprocess_image_to_tensor(img_path='image_name.jpg'):
    img = image.load_img(img_path, target_size=(IMAGE_RES, IMAGE_RES))
    x = image.img_to_array(img)/255.0                                    
    x = np.expand_dims(x, axis=0)                                              
    return x  


def save_image(item : Item):
    img_data = requests.get(item.url).content
    with open('image_name.jpg', 'wb') as handler:
        handler.write(img_data)

 
@app.post("/prediction/")
async def get_net_image_prediction(item: Item):
    if item.url == "":
        print(item.url)
        return {"message": "No image link provided"}
    
    label='name'

    save_image(item)

    x = preprocess_image_to_tensor()
    model = load_model()
    output = model.predict(x)
    prediction = np.argmax(tf.squeeze(output).numpy())

    return {'prediction': labels[label][prediction]}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000)


