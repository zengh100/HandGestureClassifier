'''
This code takes an input image to demostrate how the augmented images look like.
The training code (../train/train.py) will use the augmentation technique for model training

Example Usage:
python tools/imageDataGenerator_test.py

'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/valid/open_hand/2020.02.06_15.42.28_0047.InfraredFrame_0.png')
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
print(x.shape)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='imageDataGenerator_test', save_prefix='openhand', save_format='png'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely