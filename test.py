'''
    This file is used for testing a new image
    using a trained model
'''
import json
import cv2 as cv
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from preprocess import load_img_and_process



img = load_img_and_process('sample_image.png')
model = load_model('trained.h5', compile=False)
print(model.count_params())
model.summary()
digit = model.predict_classes(img)
print(digit[0])