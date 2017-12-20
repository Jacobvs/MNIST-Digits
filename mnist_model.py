import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.models as models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

# set seed
np.random.seed(7)


def generate_model():

    model = models.Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu', bias_initializer='RandomNormal'))
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
    print(model.summary())

    return model


def scrape_data():
    # get data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # create labels and data
    training_labels = to_categorical(train.ix[:, 0], 10)
    training_data = train.ix[:, 1:].values.reshape(train.shape[0], 28, 28, 1).astype(float)
    training_data = (25.5 + 0.8 * training_data) / 255
    testing_data = test.values.astype('float32')

    np.save('saved-files/train_labels', np.asarray(training_labels))
    np.save('saved-files/train_data', np.asarray(training_data))
    np.save('saved-files/test_data', np.asarray(testing_data))


# check to see if saved data exists, if not then create the data
if not os.path.exists('saved-files/train_labels.npy') or not os.path.exists(
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

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)

model = generate_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_labels, validation_split=0.1, epochs=3, batch_size=512)

plot_model(model, to_file='model.png', show_shapes=True)

plt.figure(1)

# summarize history for accuracy
plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(212)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save the model for later so no retraining is needed
#model.save('saved-files/model.h5')

# play sound when done with code to alert me
os.system('afplay /System/Library/Sounds/Ping.aiff')
os.system('afplay /System/Library/Sounds/Ping.aiff')









