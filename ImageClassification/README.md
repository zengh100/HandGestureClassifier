## Real-time Hand-gesture classification using Convolutional Neural Networks (CNN) on Tensorflow/Keras.

This repo documents the design and training of the hand-gesture classification model using Tensorflow/Keras.

Project Folder Structure:
. <br>
├── data <br>
│   ├── train <br>
│   │   ├── fist            # training images belong to class "fist" <br>
│   │   ├── open_hand       # training images belong to class "open_hand" <br>
│   ├── valid <br>
│   │   ├── fist            # validation images belong to class "fist" <br>
│   │   ├── open_hand       # validation images belong to class "open_hand" <br>
├── tools <br>
│   ├── imageDataGenerator_test.py: # just for testing how the augmented images look like. <br>
│   ├── k2tf_convert.py: # a tool to convert .h5 Keras model into a Tensorflow .pb file which can be deployed to embedded system that runs tensorflow C/C++ library. <br>
├── train <br>
│   ├── models   # a folder to store the output: trained model (.h5) and corresponding training metrics plots. <br>
│   ├── train.py # the source code of the model and training algorithms. inline comments and docstrings for function and modules are embedded. <br>
├── README.md    # This file. it serves as the project-level documentation.

### Data preparation:
    copy you train and valid data accordingly to folders data/train/ and data/valid/
    comments: 
        - images belong to the same class should be in the same sub-folder; the folder name represents the corresponding class. Thus the number of classes is automatically defined for the model.
        - make sure the number of images among the classes are balanced. 
### How to Train the Model:
    1. open a console
    2. enter into the root folder of this project
    3. activate your virtual environment
    4. run: python train/train.py 

### Key highlights:
    - The number of classes is easily expandable without modifying the model or training Python code.
    - Data augmentation is configured on the demand (no need during data preparation and store them as files). Simply adding more data sometimes isn't as effective as data augmentation.
    - Early stopping is adopted.

### Discussion
- When more gestures (classes) are added, the model parameters may need some adjustments.

### Potential improvements
- Add output of recalls, precisions, and confusion matrix
- The number of convolution filters and the number node in the dense layer could be adjusted (or grid search)
- Overfit Validation: this is possible. A strategy is to use a different split of the training dataset into train and validation sets each time early stopping is used.

### Refactoring
- The function train() is quite lengthy, can be further refactored. For example, the model can be defined in a separated function.

### Misc
- The training code was tested in Python 3.6.7, Tensorflow 1.12.0, and Keras 2.24