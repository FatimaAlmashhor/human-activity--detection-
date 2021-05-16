import os
import numpy as np
from keras.applications import ResNet50V2
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

# this to avoid the OSError: image file is truncated 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

height = 150
width  = 150

train_dir = './dataset/train/'
valid_dir = './dataset/validation/'
test_dir = './dataset/testing/'
classes = os.listdir(train_dir)


train_paths = []
for c in classes:
    activity_images = [train_dir+c+'/'+item for item in os.listdir(train_dir+c+'/')]
    train_paths.extend(activity_images)


valid_paths = []
for c in classes:
    activity_images = [valid_dir+c+'/'+item for item in os.listdir(valid_dir+c+'/')]
    valid_paths.extend(activity_images)

test_paths = []
for c in classes:
    activity_images = [test_dir+c+'/'+item for item in os.listdir(test_dir+c+'/')]
    test_paths.extend(activity_images)

def preprocess_image(path):
    img = image.load_img(path, target_size = (height, width))
    a = image.img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    return preprocess_input(a)
    
train_preprocessed_images = np.vstack([preprocess_image(fn) for fn in train_paths])
np.save("train_preprocesed.npy",train_preprocessed_images)

valid_preprocessed_images = np.vstack(preprocess_image(fn) for fn in valid_paths)
np.save("valid_preprocessed.npy",valid_preprocessed_images)

test_preprocessed_images = np.vstack(preprocess_image(fn) for fn in test_paths)
np.save("test_preprocessed.npy",test_preprocessed_images)
