# TF2 version
#import tensorflow.compat.v2 as tf
#import tensorflow_hub as hub

from fastapi import FastAPI

#model = hub.KerasLayer('https://www.kaggle.com/models/google/aiy/frameworks/TensorFlow1/variations/vision-classifier-birds-v1/versions/1')

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Welcome!'}

