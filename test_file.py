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

################################## image preprocessing ###########################################

############################## now making the attention U-net ####################################

#Build the model
inputs = tf.keras.layers.Input((PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x/255)(inputs) #for x devide x by 255 and x corresponds to inputs
# this keras.lambds is same as the regular lambda function in regular python.
# where in regular python instead of defining as regular function by def we can use the
# lambda function to unwrap the function within that line.

# Contraction Path
c1 = tf.keras.layers.Conv2D(128, (9,9), activation='relu', padding='same')(s)
c2 = tf.keras.layers.Conv2D(64, (7,7), activation='relu', padding='same')(c1)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c2)
c4 = tf.keras.layers.Conv2D(32, (1,1), activation='relu', padding='same')(c3)
# testing changes are to be made from here
outputs = tf.keras.layers.Conv2D(3, (5,5), activation='softmax', padding='same')(c4)
#try selu
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
# try optimizer SGD
# since we will be working with a binary classification i.e. either a cell or not a cell so we will be using
# binary_crossentropy as our loss function.
model.summary()
