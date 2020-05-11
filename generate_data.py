import numpy as np
import os

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize

PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_CHANNELS = 3

ORG_WIDTH = 2048
ORG_HEIGHT = 1536
ORG_CHANNELS = 3

# Training Dataset
print("Processing For Training Dataset !")
TRAIN_PATH = "dataset/train/"

X_TRAIN_PATH = TRAIN_PATH +  "defocused_blurred/"
Y_TRAIN_PATH = TRAIN_PATH +  "sharp/"

x_ids_train = next(os.walk(X_TRAIN_PATH))[2]
y_ids_train = next(os.walk(Y_TRAIN_PATH))[2]

x_train = np.zeros((len(x_ids_train), ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), dtype=np.uint8)
y_train = np.zeros((len(y_ids_train), ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), dtype=np.uint8)

for n,img in tqdm(enumerate(x_ids_train), total=len(x_ids_train)):
    image = imread(X_TRAIN_PATH + img)
    image = resize(image, (ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), mode='constant', preserve_range=True)
    x_train[n] = image

for n,img in tqdm(enumerate(y_ids_train), total=len(y_ids_train)):
    image = imread(Y_TRAIN_PATH + img)
    image = resize(image, (ORG_WIDTH, ORG_HEIGHT, ORG_CHANNELS), mode='constant', preserve_range=True)
    y_train[n] = image

print("converting into patches !")
# converting into patches -->
# 192 patches from each images
# so 3 batches of size = 64 will make 1 image complete
# total images = 192*325 = 62400

X_train = np.zeros((len(x_train), 192, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(y_train), 192, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.uint8)

for n, img in tqdm(enumerate(x_train), total=len(x_train)):
    img = x_train[n]
    M,N,O = img.shape
    b0,b1,b2 =128,128,3
    x = img.reshape(M//b0,b0,N//b1,b1,b2).swapaxes(1,2).reshape(-1,b0,b1,b2)
    X_train[n] = x

X_train = X_train.reshape(-1, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS)


for n, img in tqdm(enumerate(y_train), total=len(y_train)):
    img = y_train[n]
    M,N,O = img.shape
    b0,b1,b2 =128,128,3
    x = img.reshape(M//b0,b0,N//b1,b1,b2).swapaxes(1,2).reshape(-1,b0,b1,b2)
    Y_train[n] = x

Y_train = Y_train.reshape(-1, PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS)

print("Normalizing Y_train !")
Y_train_new = np.zeros((len(Y_train), PATCH_WIDTH, PATCH_HEIGHT, PATCH_CHANNELS), dtype=np.float32)

for n, img in tqdm(enumerate(Y_train), total=len(Y_train)):
    Y_train_new[n] = Y_train[n]/255

print("saving data !")
path_np = "np_folder/"
np.save(path_np+"X_train", X_train)
np.save(path_np+"Y_train", Y_train_new)
