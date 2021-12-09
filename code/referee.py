import os
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, MaxPooling2D, Dropout, Conv2D, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.applications import EfficientNetB0

class Referee(tf.keras.Model):
    def __init__(self, batch_size) -> None:
        super(Referee, self).__init__()

        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.001
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = batch_size
        self.conv_size = 128
        self.hidden_size = 400

        # TODO: Initialize all trainable parameters

        self.referee = Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(self.conv_size, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(self.hidden_size, activation='relu'),
            Dense(self.hidden_size, activation='relu'),
            Dense(100, activation='softmax')
        ])

    def call(self, inputs):
        # TODO: Write forward-pass logic
        # outputs a 1 by 1 by 20
        referee_output = self.referee(inputs)
        return referee_output
    
    def loss(self, probs, labels): 
        # TODO: Write loss function

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        return tf.reduce_mean(loss)
    