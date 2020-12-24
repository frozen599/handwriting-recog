'''
Implement a function for loading and processing 
an image before it could be used for testing a trained model
'''

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def load_img_and_process(path: str) -> np.ndarray:
    img = load_img(path, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = np.array([img])
    img = img.astype('float32')
    img = img / 255.0
    return img


