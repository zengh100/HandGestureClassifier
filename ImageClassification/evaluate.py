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
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis = 0)
    #print(img.shape)
    return img

def predict_examples():
    # Load model from previously saved model file fromtraining
    print('loading... from the best weight model')
    filepath = './cnn/ir_preprocessed/models_2classes_batch=128_epoch=1000_lr=0.001_decay=5e-07_dense=128/' + 'weights.best.hdf5'
    model = keras.models.load_model(filepath) # load from the best saved model

    # Evaluation
    images = []
    image_folder = './data_2classes_ir_preprocessed/valid/fist/'
    image_filepath = image_folder+'2019.10.04_15.36.02_0010.InfraredFrame_0.png'
    image = load_keras(image_filepath)
    images.append(image)
    image_concatenated = image

    image_folder = './data_2classes_ir_preprocessed/valid/open_hand/'
    image_filepath = image_folder+'2019.10.04_15.23.01_0007.InfraredFrame_0.png'
    image = load_keras(image_filepath)
    images.append(image)
    image_concatenated = np.concatenate((image_concatenated, image))

    for image in images:
        class_prob = model.predict(image)
        print(class_prob)

    print(image_concatenated.shape)
    class_prob = model.predict(image_concatenated)
    print(class_prob)

def predict_folder(model, image_folder, label_index, num_class):
    #search sub-folder
    search_results = glob.glob(image_folder+'*.png')
    if len(search_results) == 0: 
        print('No image found!')

    N = len(search_results)
    print("searched search_results=", N)
    
    correct_dict = {'fist':0, 'open_hand':0}

    predict_list = [0]*num_class
    for index,filename in enumerate(search_results):
        image = load_keras(filename)
        class_prob = model.predict(image)
        #print(class_prob[0])
        ind = np.unravel_index(np.argmax(class_prob[0], axis=None), class_prob[0].shape)
        #print('detected class inex is ', ind[0])
        predict_list[ind[0]] += 1

        #result = np.where(class_prob[0] == np.amax(class_prob[0]))
        #print('detected class inex is ', result[0][0])

        '''       
        if(class_prob[0,label_index] > 0.5): 
            correct[index] += 1
        else:
            print(index, filename)
            print(class_prob)
        '''
    print(f"Accuracy = {predict_list[label_index]/N} : {int(predict_list[label_index])}/{N}.")
    return predict_list,N


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    #predict_examples()

    data_folder = './data/'
    model_folder = './train/models/'    
    model_filepath =  model_folder + 'weights.min.val_loss.hdf5'
    #model_filepath =  model_folder + 'batch=128_epoch=500_dense=128.h5'
    #model_filepath =  model_folder + 'batch=128_epoch=1000_dense=128.h5'
    #model_filepath =  model_folder + 'batch=128_epoch=1000_dense=128_scale=2500.h5'
    #model_filepath =  model_folder + 'batch=128_epoch=1000_dense=128_scale=255.h5'

    # Load model from previously saved model file fromtraining
    print(f'loading... model from {model_filepath}')
    model = keras.models.load_model(model_filepath) # load from the best saved model

    statistics_list = []

    targets = ['valid', 'train']
    labels=['fist', 'open_hand']
    for index, target in enumerate(targets):
        labels = get_subDirList(data_folder+target+'/')
        print('labels=', labels)
        if(index>0):
            #compare with previous labels
            if len(labels) != len(last_labels):
                print("number of classes is different")
                quit()
            common = [i for i, j in zip(labels, last_labels) if i == j]
            if(len(common) != len(labels)):
                print('number of common subfolders is less: ', common)
                quit()
        last_labels = labels
    
    column_name = ['class', 'train/test', 'number', 'correct', 'accuracy']
    column_name += labels
    print(f'column_name={column_name}')

    total_predict_list = [0]*len(labels)
    correct_list = [0]*len(labels)
    total_correct = 0
    N = 0
    for target in targets:
        for index,label in enumerate(labels):        
            predict_list,N_ = predict_folder(model, data_folder+target+'/'+label+'/',index, len(labels))
            value = (label, target, N_, predict_list[index], predict_list[index]/N_)
            value = value + tuple(predict_list)
            statistics_list.append(value)
            total_predict_list = [sum(x) for x in zip(total_predict_list, predict_list)]
            total_correct += predict_list[index]
            correct_list[index] += predict_list[index]
            N += N_

    #correct_,N_ = predict_folder('./data_2classes_ir_preprocessed/valid/fist/', 0, 2)
    '''
    correct_,N_ = predict_folder(0, './data_2classes_ir_preprocessed/valid/fist/')
    value = ('fist', 'test', N_, correct_, correct_/N_, correct_, N_-correct_)
    statistics_list.append(value)
    correct += correct_
    N += N_

    correct_,N_ = predict_folder(0, './data_2classes_ir_preprocessed/train/fist/')
    value = ('fist', 'train', N_, correct_, correct_/N_, correct_, N_-correct_)
    statistics_list.append(value)
    correct += correct_
    N += N_
    correct_,N_ = predict_folder(1, './data_2classes_ir_preprocessed/valid/open_hand/')
    value = ('open_hand', 'test', N_, correct_, correct_/N_, N_-correct_, correct_)
    statistics_list.append(value)
    correct += correct_
    N += N_
    correct_, N_ = predict_folder(1, './data_2classes_ir_preprocessed/train/open_hand/')
    value = ('open_hand', 'train', N_, correct_, correct_/N_, N_-correct_, correct_)
    statistics_list.append(value)
    correct += correct_
    N += N_
    print(f"Total Accuracy = {correct/N} : {int(correct)}/{N}.")

    '''
    value = ('total', 'both', N, total_correct, total_correct/N)
    value = value + tuple(total_predict_list)
    statistics_list.append(value)

    #print('\ncorrect_list\n', correct_list)
    #print('\ntotal_predict_list\n', total_predict_list)
    precision_list = [i / j for i, j in zip(correct_list, total_predict_list)] 
    value = ('precison', '---', '---', '---', '---')
    value = value + tuple(precision_list)
    statistics_list.append(value)

    stat_df = pd.DataFrame(statistics_list, columns=column_name)
    output_csv_filename = model_folder + get_filename_noExt(model_filepath) +'_training_stat.csv'
    print("output_csv_filename=" + output_csv_filename)
    stat_df.to_csv(output_csv_filename, index = None)
