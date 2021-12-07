import os
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.applications.vgg16 import VGG16

class Referee(tf.keras.Model):
    def __init__(self, batch_size) -> None:
        super(Referee, self).__init__()

        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.01
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = batch_size
        # TODO: Initialize all trainable parameters
        self.referee = Sequential([
            VGG16(include_top=False, 
            weights="imagenet"),
            Dense(4096, activation="relu"),
            Dense(4096, activation="relu"),
            Dense(20, activation="relu")
        ])
        pass

    def call(self, inputs):
        # TODO: Write forward-pass logic
        # outputs a 1 by 1 by 20 , normalized
        refereeOutput = tf.nn.softmax(self.referee(inputs))

        # this returns probability of input being one of the 20 images
        return refereeOutput
    
    def loss(self, probs, labels): 
        # TODO: Write loss function
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        return tf.reduce_sum(loss)
    