from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, \
    add, multiply, Conv2DTranspose
from keras.layers import concatenate, core, Dropout
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
from keras.activations import linear as l
import keras.backend as K
import keras as k

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

def attention_down_and_concate(up_layer, layer, data_format = data_format):
    if data_format == 'channel_first':
        in_channel = up_layer.get_shape().as_list()[1]
    else:
        in_channel = up_layer.get_shape().as_list()[3]

    down = MaxPooling2D((2,2), data_format=data_format)(up_layer)
    layer = attention_block_2d(x=layer, g=down, inter_channel=in_channel//4, data_format = data_format)

    if data_format == 'channel_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate =  my_concat([down, layer])

    return concate


def attention_block_2d(x, g, inter_channel, data_format=data_format):
    theta_x = Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(x)

    phi_g = Conv2D(inter_channel, [1,1], strides=[1,1], data_format=data_format)(g)

    f = Activation('relu')(add([theta_x,phi_g]))

    psi_f = Conv2D(1, [1,1], strides=[1,1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x,rate])

    return att_x



################################## image preprocessing ###########################################
print("Loading training dataset !")
print("X_train")
X_train = np.load("np_folder/X_train.npy")
print("Y_train")
Y_train = np.load("np_folder/Y_train.npy")
print("Data Loaded !")
############################## now making the attention inverted U-net ####################################

inputs = Input((PATCH_WIDTH,PATCH_HEIGHT,PATCH_CHANNELS))
x = Lambda(lambda x: x/255)(inputs)
depth = 2
# try with depth 3 and 4
features = 16
# try features 32 also
num_class = 3
skips = []

for i in range(depth):
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    skips.append(x)
    x = UpSampling2D(size = (2,2), data_format=data_format)(x)
    #x = Conv2DTranspose(features, (2,2), strides = (2,2), name = 'up', padding = 'same')(x)
    features = features*2

x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
x = Dropout(0.2)(x)
x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)

for i in reversed(range(depth)):
    features = features//2
    x = attention_down_and_concate(x, skips[i], data_format=data_format)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3,3), activation='relu', padding='same', data_format=data_format)(x)
# cahnges to be made down here
conv6 = Conv2D(num_class, (1,1), activation='softmax', padding='same', data_format=data_format)(x)
output = l(conv6)
#output = core.Activation('elu')(conv6)
# try selu
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
# mean_squared_error try this error function
# adam is giving average accuracy of 45%
model.summary()

########################################################################################################

callbacks = [
    k.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    k.callbacks.TensorBoard(log_dir='logs\{}'.format(NAME)),
    k.callbacks.ModelCheckpoint('test_model_new2.h5', verbose=1, save_best_only=True)]

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=3, epochs=25, verbose=1, callbacks=callbacks)

# to run the tensorboard command
# tensorboard --logdir=logs/
# now run the command and the oprn the browser
# you will get a local host address copy that address
# in the browser address bar type following:
# pat the copied address
# and press enter
