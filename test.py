import os
import time
import numpy as np
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

# Flatten the dataset
train_images = train_images.reshape(TRAINING_SIZE, -1).astype(np.float32)
test_images = test_images.reshape(TEST_SIZE, -1).astype(np.float32)

# Convert pixel value from integers between 0 and 255 to floats between 0 and 1
train_images /= 255
test_images /= 255

NUM_DIGITS = 10
print('Before', train_labels[0])
train_labels = keras.utils.to_categorical(train_labels, NUM_DIGITS)
print('After', train_labels[0])
test_labels = keras.utils.to_categorical(test_labels, NUM_DIGITS)

model = keras.Sequential()
model.add(keras.layers.Dense(512, activation="relu", input_shape=(784,)))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

model.fit(train_images, train_labels, epochs=5)
loss, accuracy = model.evaluate(test_images, test_labels)
print(loss, accuracy)