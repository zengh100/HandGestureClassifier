import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

# prepare to train
"""
conda info --envs
conda activate py36_tf1131_gpu
if missing PIL run below
pip install pillow
"""

# How to run this script
# python train/train.py
#
def plot_metrics(hist, savefig_filepath):
    """plot post-training metrics and save the plots to a image file.
    Args:
        hist (object): the history returned from model fitting
        savefig_filepath (str): the file path to save the metrics plots.
    Returns:
        None
    """    
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.savefig(savefig_filepath)
    plt.show()

def train():
    """Train the classification model.
    Args:
        None
    Returns:
        None
    """    

    """Training parameters including hyper parameters"""
    batch_size = 128
    epochs = 1000 #it might stop earlier since EarlyStopping technique is adopted
    epochs_max = 2000 # to control decay
    learning_rate = 0.001
    decay = learning_rate/epochs_max
    rescale_deno = 255 #always for ir images
    #rescale_deno = 2500 # consider for depth images
    print(f'batch_size={batch_size}, epochs={epochs}, learning_rate={learning_rate}, decay={decay}, rescale_deno={rescale_deno}')

    # input image dimensions
    img_rows, img_cols = 46, 46

    rescale = 1./rescale_deno 
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            rescale=rescale,
            shear_range=0.2,
            zoom_range=0.2,
            #brightness_range=[0.2,1.0],
            horizontal_flip=True
            ) 
    # this is the augmentation configuration we will use for validating:
    # only rescaling
    valid_datagen = ImageDataGenerator(rescale=rescale)

    # the data, shuffled and split between train and test sets
    #x_train, y_train, x_test, y_test = dataset.load_data(poses=["all"])
    print('\nLoading train data')
    train_generator = train_datagen.flow_from_directory(
        directory=r"data/train/",
        target_size=(img_rows, img_cols),
        color_mode="grayscale", #rgb
        batch_size=128,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    print('train data class_indices:')
    print(train_generator.class_indices)

    print('\nLoading valid data')
    valid_generator = valid_datagen.flow_from_directory(
        directory=r"data/valid/",
        target_size=(img_rows, img_cols),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    num_classes = train_generator.num_classes
    print('num_classes=', num_classes)

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    print('input_shape:', input_shape)

    ####### Model structure #######
    #model building
    model = Sequential()
    # convolutional layer with rectified linear unit activation
    # 32 convolution filters used each of size 3x3
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    # again
    # 64 convolution filters used each of size 3x3
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    # flatten before adding a dense layer
    model.add(Flatten())
    # fully connected to get all relevant data
    num_node_denseLayer = 128
    model.add(Dense(num_node_denseLayer, activation='relu'))

    # another dropout for better convergence 
    model.add(Dropout(0.5))
    # output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_classes, activation='softmax'))
    # categorical cross entropy since we have multiple classes
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(lr=learning_rate, decay=decay),
                metrics=['accuracy'])

    """ callbacks """
    # checkpoint
    monitor = 'val_loss' # if use 'val_acc' then mode should be 'max'
    mode = 'min'
    best_model_filepath=f"train/models/weights.{mode}.{monitor}.hdf5"
    #best_model_filepath="train/models/weights.{epoch:04d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(best_model_filepath, monitor=monitor, verbose=1, save_best_only=True, mode=mode)
    # early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    callbacks_list = [checkpoint, es]

    #Fitting/Training the model
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    hist = model.fit_generator(generator=train_generator, \
                        steps_per_epoch=STEP_SIZE_TRAIN,  \
                        validation_data=valid_generator,  \
                        validation_steps=STEP_SIZE_VALID, \
                        callbacks=callbacks_list,         \
                        epochs=epochs                     \
    )
    print(f'stopped_epoch={es.stopped_epoch}') 

    # Evaluation
    score = model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model_name = f"train/models/batch={batch_size}_epoch={es.stopped_epoch}_dense={num_node_denseLayer}_scale={rescale_deno}.h5"
    savefig_filepath = f"train/models/batch={batch_size}_epoch={es.stopped_epoch}_dense={num_node_denseLayer}_scale={rescale_deno}_lr={learning_rate}_decay={decay}.png"
    model.save(model_name)

    print(f'batch_size={batch_size}, epochs={es.stopped_epoch}, learning_rate={learning_rate}, decay={decay}, num_node_denseLayer={num_node_denseLayer}, num_classes={num_classes}, scale={rescale_deno}')

    time.sleep(1)

    # plotting the metrics
    plot_metrics(hist, savefig_filepath)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    train()
