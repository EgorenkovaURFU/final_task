import tensorflow_hub as hub
import tensorflow as tf

from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel, HttpUrl
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.efficientnet import decode_predictions

from rembg import remove 
from PIL import Image 

import pandas as pd

IMAGE_RES = 224 

labels = pd.read_csv("labels_oiseaux.csv", sep=';', header=0, index_col=0)


class Item(BaseModel):
    url: HttpUrl
    # str: str


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


# def preprocess_image_to_tensor(item: Item):
#     img = image.load_img(get_file('image', str(item.url)), target_size=(IMAGE_RES, IMAGE_RES))
#     x = image.img_to_array(img)/255.0                                           
#     x = np.expand_dims(x, axis=0)                                              
#     return x

# def remove_background(item: Item):
#     #img = image.load_img(get_file('image', str(item.url)), target_size=(IMAGE_RES, IMAGE_RES))
#     img = get_file('image', str(item.url))
#     input = Image.open(img)
#     output = remove(input)
#     return output


def preprocess_image_to_tensor(img_path):
    img = image.load_img(img_path, target_size=(IMAGE_RES, IMAGE_RES))
    x = image.img_to_array(img)/255.0                                    
    x = np.expand_dims(x, axis=0)                                              
    return x  
   

@app.post("/prediction/")
async def get_net_image_prediction(item: Item):
    if item.url == "":
        print(item.url)
        return {"message": "No image link provided"}
    
    label='name'

    print(type(output))
    # x = preprocess_image_to_tensor(item.str)
    # model = load_model()
    # output = model.predict(x)
    # prediction = np.argmax(tf.squeeze(output).numpy())

    # return {'prediction': labels[label][prediction]}


