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
        self.hidden_size = 200

        # TODO: Initialize all trainable parameters

        self.VGG = VGG16(include_top=False, 
            weights="imagenet")

        self.referee = Sequential([
            Flatten(),
            Dense(self.hidden_size, activation="relu"),
            Dense(self.hidden_size, activation="relu"),
            Dense(80, activation="relu")
        ])
        pass

    def call(self, inputs):
        # TODO: Write forward-pass logic
        # outputs a 1 by 1 by 20 , normalized

        vgg_output = self.VGG(inputs)

        print(vgg_output.shape)

        referee_output = tf.nn.softmax(self.referee(vgg_output))

        # this returns probability of input being one of the 20 images
        return referee_output
    
    def loss(self, probs, labels): 
        # TODO: Write loss function
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=False)
        return tf.reduce_sum(loss)
    