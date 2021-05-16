import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import EarlyStopping, History, ModelCheckpoint
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.normalization import BatchNormalization

train_dir = './dataset/train/'
valid_dir = './dataset/validation/'

# call the iamges after expand_dims 
train_features = np.load('train_preprocesed.npy')
valid_features = np.load('valid_preprocessed.npy')

classes = os.listdir(train_dir)


# Get the labels
# named all the images as its class name
train_labels = []
for c in classes:
    l = [c]*len(os.listdir(train_dir+c+'/'))
    train_labels.extend(l)


valid_labels = []
for c in classes:
    l = [c]*len(os.listdir(valid_dir+c+'/'))
    valid_labels.extend(l)


onehot_train = to_categorical(LabelEncoder().fit_transform(train_labels))
onehot_valid = to_categorical(LabelEncoder().fit_transform(valid_labels))

vgg = VGG16(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(150, 150,3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding new layers...')
output = vgg.get_layer(index = -1).output  
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(4096,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
# and a logistic layer -- let's say we have 200 classes
output = Dense(6, activation='softmax')(output)

vgg_model = Model(vgg.input, output)
vgg_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =["accuracy"])
# print(vgg_model.summary())


train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

val_datagen = ImageDataGenerator()
callbacks = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')        
# autosave best Model
best_model_file = "./data_augmented_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

history = vgg_model.fit_generator(train_datagen.flow(train_features, onehot_train, batch_size=25), nb_epoch=20,
              samples_per_epoch = 4243,                     
              validation_data=val_datagen.flow(valid_features,onehot_valid,batch_size=10,shuffle=False),
                                    nb_val_samples=282,callbacks = [callbacks,best_model])

# vgg_model.load_weights("data_augmented_weights.h5")
# summarize history for accuracy
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch'); plt.legend(['train', 'valid'], loc='upper left')

# summarize history for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss'); plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()