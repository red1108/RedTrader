'''
https://github.com/quantylab/rltrader/blob/master/networks.py 
'''


import os

import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


class Network:

    def __init__(self, input_dim=0, output_dim=0, lr=0.001, 
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.shared_network = shared_network
        self.activation = activation
        self.loss = loss
        self.model = None

    def predict(self, sample):
        return self.model(sample).flatten()

    def train_on_batch(self, x, y):
        loss = 0.
        history = self.model.fit(x, y, epochs=10, verbose=False)
        loss += np.sum(history.history['loss'])
        return loss

    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True)

    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)



class DNN(keras.Model):
    def __init__(self, activation, output_dim):
        #keras.backend.set_floatx('float32')
        super().__init__()
        self.activation = activation
        
        self.dense1 = Dense(256, activation=self.activation,
                            kernal_initializer='random_normal')
        self.dense2 = Dense(128, activation=self.activation,
                            kernal_initializer='random_normal')
        self.dense3 = Dense(64, activation=self.activation,
                            kernal_initializer='random_normal')
        self.dense4 = Dense(32, activation=self.activation,
                            kernal_initializer='random_normal')
        self.dense5 = Dense(output_dim, activation=self.activation,
                            kernal_initializer='random_normal')
        self.drop1 = Dropout(0.1)
        self.drop2 = Dropout(0.1)
        self.drop3 = Dropout(0.1)
        self.drop4 = Dropout(0.1)
    
    def call(self, x) :
        x = self.dense1(x)
        x = self.drop1(x)
        x = self.dense2(x)
        x = self.drop2(x)
        x = self.dense3(x)
        x = self.drop3(x)
        x = self.dense4(x)
        x = self.drop4(x)
        x = self.dense5(x)
        return x



