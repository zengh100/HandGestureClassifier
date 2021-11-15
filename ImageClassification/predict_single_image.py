import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os,sys
import glob
import pandas as pd

import yaml

from preprocessing import (
    read_img_from_path,
    resize_img,
    read_from_file,
)
from util import download_model

# How to run this script
# python ./predict.py
#
def get_subDirList(folder):
    import os
    subFolderList = [dI for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder,dI))]
    return subFolderList	

def get_filename_noExt(filePath):
    from pathlib import Path
    return Path(filePath).stem

def writeDictToCSV(dict, csv_filename):
    with open(csv_filename, 'w', newline="") as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])

'''
def load(filename):
    from PIL import Image
    from skimage import transform
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (46, 46, 1))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image
'''

def load_keras(filename):
    from keras.preprocessing import image
    img_width, img_height = 46, 46
    img = image.load_img(filename, target_size = (img_width, img_height), color_mode='grayscale')
    # After keras.preprocessing.image.load_img(), the img is of type <class 'PIL.Image.Image'>, and it doesn't support the use of .shape
    # but it can be converted to numpy.ndarray
    # print("type of image from   keras.preprocessing.image.load_img = ", type(img))
    #print("shape of image from   keras.preprocessing.image.load_img = ", img.shape) #not supported to use .shape
    img = image.img_to_array(img)
    # After image.img_to_array(), the img is of type numpy.ndarray, and the shape is 46x46x1
    #print("shape of image after image.img_to_array = ", img.shape)
    print("type of image after image.img_to_array = ", type(img))
    print('image.dtype=', img.dtype)
    img /= 255.0
    img = np.expand_dims(img, axis = 0)
    # After np.expand_dims(img, axis = 0), img has a shape of (1, 46, 46, 1)
    # print("shape of image after np.expand_dims(img, axis = 0) = ", img.shape)

    print("max value = ", img.max())
    print("min value = ", img.min())
    print('image.dtype=', img.dtype)

    return img

class ImagePredictor:
    def __init__(
        self, model_path, resize_size, targets#, pre_processing_function=preprocess_input
    ):
        self.model_path = model_path
        #self.pre_processing_function = pre_processing_function
        print("Loading model from file: ", self.model_path)
        self.model = keras.models.load_model(self.model_path)
        self.resize_size = resize_size
        self.targets = targets
    @classmethod
    def init_from_config_path(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        predictor = cls(
            model_path=config["model_path"],
            resize_size=config["resize_shape"],
            targets=config["targets"],
        )
        return predictor
    @classmethod
    def init_from_config_url(cls, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        download_model(
            config["model_url"], config["model_path"], config["model_sha256"]
        )
        return cls.init_from_config_path(config_path)
    
    def predict_from_array(self, arr):
        arr = resize_img(arr, h=self.resize_size[0], w=self.resize_size[1])
        # print('image_after_resize.shape=', arr.shape)
        # print("type of image_after_resize=", type(arr)  )
        # print("dtype of image_after_resize=", arr.dtype  )
        #arr = self.pre_processing_function(arr)
        #arr = np.float32(arr) # not necessary here since the image was read astype(np.float32)
        arr /= 255.0 # This is allowed since it is float32 dtype
        # print("dtype of image_after_resize=", arr.dtype  )
        # print("max value = ", arr.max())
        # print("min value = ", arr.min())
        # The model requires shape of (n,46,46,1)
        arr = arr[np.newaxis, ..., np.newaxis] # This makes shape of (1,46,46,1)
        #print('image_before_predict.shape=', arr.shape)
        pred = self.model.predict(arr)
        #print("pred=", pred)
        pred = pred.ravel().tolist()
        #print("pred_converted =", pred)
        pred = [round(x, 3) for x in pred]
        return {k: v for k, v in zip(self.targets, pred)}

    def predict_from_file(self, file_object): #This is for url file
        arr = read_from_file(file_object)
        return self.predict_from_array(arr)

    def predict_from_path(self, path):
        arr = read_img_from_path(path)
        print('image_from_path.shape=', arr.shape)
        print('image_from_path.type=', type(arr))
        print('image_from_path.dtype=', arr.dtype)

        return self.predict_from_array(arr)

def predict_examples():
    # Load model from previously saved model file fromtraining
    print('loading... from the best weight model')
    data_folder = './data/'
    model_folder = './train/models/'    
    model_filepath =  model_folder + 'weights.min.val_loss.hdf5'    
    model = keras.models.load_model(model_filepath) # load model from file

    # Evaluation
    images = []
    image_folder = './data/valid/fist/'
    image_filepath = image_folder+'2019.10.04_15.36.02_0010.InfraredFrame_0.png'
    image = load_keras(image_filepath)
    images.append(image)
    image_concatenated = image

    image_folder = './data/valid/open_hand/'
    image_filepath = image_folder+'2019.10.04_15.23.01_0007.InfraredFrame_0.png'
    image = load_keras(image_filepath)
    images.append(image)
    image_concatenated = np.concatenate((image_concatenated, image))

    for image in images:
        print('image.shape: ', image.shape)
        class_prob = model.predict(image)
        print(class_prob)

    print('\nimage_concatenated.shape: ', image_concatenated.shape)
    class_prob = model.predict(image_concatenated)
    print(class_prob)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    #predict_examples()

    print('------------------------------------')
    print("Use of class ImagePredictor - predict_from_path")
    model_folder = './train/models/'    
    model_filepath =  model_folder + 'weights.min.val_loss.hdf5'  
    targets = ["fist", "open_hand"]
    predictor = ImagePredictor(model_filepath, (46,46), targets)

    predictor_config_path = "config.yaml"
    predictor.init_from_config_path(predictor_config_path)

    image_folder = './data/valid/open_hand/'
    image_filepath = image_folder+'2019.10.04_15.23.01_0007.InfraredFrame_0.png'
    class_prob = predictor.predict_from_path(image_filepath)
    print(class_prob)

    print('------------------------------------')
    print("Use of class ImagePredictor - predict_from_file ")
    with open(image_filepath, "rb") as f:
        class_prob = predictor.predict_from_file(f)
        print(class_prob)




