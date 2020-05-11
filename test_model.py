import tensorflow as tf

import numpy as np
import os
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

seed = 99
np.random.seed = seed

NAME = "Test-Net_CNN-{}".format(int(time.time()))

data_format = 'channels_last'

# actually the size of the IMAGE patch
PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_CHANNELS = 3

ORG_WIDTH = 2048
ORG_HEIGHT = 1536
ORG_CHANNELS = 3


#Test Dataset
print("For Test Dataset !")

TEST_PATH = "dataset/test/"

X_TEST_PATH = TEST_PATH +  "defocused_blurred/"
Y_TEST_PATH = TEST_PATH +  "sharp/"

x_ids_test = next(os.walk(X_TEST_PATH))[2]
y_ids_test = next(os.walk(Y_TEST_PATH))[2]

x_test = np.zeros((len(x_ids_test), ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), dtype=np.uint8)
y_test = np.zeros((len(y_ids_test), ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), dtype=np.uint8)

for n,img in tqdm(enumerate(x_ids_test), total=len(x_ids_test)):
    image = imread(X_TEST_PATH + img)
    image = resize(image, (ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), mode='constant', preserve_range=True)
    x_test[n] = image

for n,img in tqdm(enumerate(y_ids_test), total=len(y_ids_test)):
    image = imread(Y_TEST_PATH + img)
    image = resize(image, (ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), mode='constant', preserve_range=True)
    y_test[n] = image


X_test = np.zeros((len(x_test), 192, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(y_test), 192, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.uint8)

for n,img in tqdm(enumerate(x_test), total=len(x_test)):
    img = x_test[n]
    M,N,O = img.shape
    b0,b1,b2 = 128,128,3
    x = img.reshape(M//b0,b0,N//b1,b1,b2).swapaxes(1,2).reshape(-1,b0,b1,b2)
    X_test[n] = x

X_test = X_test.reshape(-1, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS)

for n,img in tqdm(enumerate(y_test), total=len(y_test)):
    img = y_test[n]
    M,N,O = img.shape
    b0,b1,b2 = 128,128,3
    x = img.reshape(M//b0,b0,N//b1,b1,b2).swapaxes(1,2).reshape(-1,b0,b1,b2)
    Y_test[n] = x

Y_test = Y_test.reshape(-1, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS)

Y_test_new = np.zeros((len(Y_test), PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.float64)

for n, img in tqdm(enumerate(Y_test), total=len(Y_test)):
    Y_test_new[n] = Y_test[n]/255


####
model = tf.keras.models.load_model("test_model.h5")

model.summary()

img = X_test[1]
img = np.expand_dims(img, axis=0)

pred_test = model.predict(img, verbose = 1)
# now convert pred_test to normal scale by multiplying it with 255 and rounding off to top or bottom.

imshow(X_test[1])
plt.show()
#imshow(pred_test)
#plt.show()
imshow(Y_test[1])
plt.show()
