import os
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.math import exp, sqrt, square
from tensorflow.keras.applications import ResNet50V2

class Referee(tf.keras.Model):
    def __init__(self, batch_size) -> None:
        super(Referee, self).__init__()

        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.01
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.batch_size = batch_size
        self.hidden_size = 200

        # TODO: Initialize all trainable parameters

        self.ResNet50 = ResNet50V2(include_top=False, 
            weights="imagenet")

        self.ResNet50.trainable = False

        self.referee = Sequential([
            Flatten(),
            Dense(self.hidden_size, activation="relu"),
            Dense(self.hidden_size, activation="relu"),
            Dense(100, activation="relu")
        ])
        pass

    def call(self, inputs):
        # TODO: Write forward-pass logic
        # outputs a 1 by 1 by 20
        print(inputs.shape)
        resnet_output = self.ResNet50(inputs)
        print(resnet_output.shape)
        referee_output = self.referee(resnet_output)

        return referee_output
    
    def loss(self, probs, labels): 
        # TODO: Write loss function
        print(probs.shape)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs, from_logits=True)
        return tf.reduce_sum(loss)
    