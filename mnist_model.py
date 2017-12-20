import os
import numpy as np
import pandas as pd
import keras.models as models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils.np_utils import to_categorical

# set seed
np.random.seed(7)


def generate_model():

    model = models.Sequential()

    model.add(Conv2D(1, (5, 5), input_shape=(1, 28, 28), activation='relu', bias_initializer='RandomNormal'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation="softmax"))

    return model



def scrape_data():
    # get data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # create labels and data
    train_label = train.ix[:, 0].values.astype('int32')
    train_label = to_categorical(train_label, 10)
    train_data = train.ix[:, 1:].values.reshape(train.shape[0], 1, 28, 28).astype('float32')
    test_data = test.values.astype('float32')

    np.save('saved-files/train_label', np.asarray(train_label))
    np.save('saved-files/train_data', np.asarray(train_data))
    np.save('saved-files/test_data', np.asarray(test_data))


# check to see if saved data exists, if not then create the data
if not os.path.exists('saved-files/train_label.npy') or not os.path.exists(
        'saved-files/train_data.npy') or not os.path.exists(
        'saved-files/test_data.npy'):
    print('Creating Data')
    if not os.path.exists('saved-files'):
        os.mkdir('saved-files')
    scrape_data()

#load data
train_data = np.load('saved-files/train_data.npy')
train_labels = np.load('saved-files/train_labels.npy')
test_data = np.load('saved-files/test_data.npy')










