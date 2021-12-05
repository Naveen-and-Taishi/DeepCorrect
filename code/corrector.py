import os
import tensorflow as tf
import numpy as np
import random
import math

from simulator import Simulator

class Corrector(tf.keras.Model):
    def __init__(self, batch_size, type) -> None:
        super(Corrector, self).__init__()

        self.learning_rate = 0.01
        self.Adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.batch_size = batch_size

        # maybe add max pool layers ?

        self.linear_corrector = tf.convert_to_tensor([[0, 0, 0], [0.7, 1, 0], [0.7, 0, 1]])

        self.conv1 = tf.keras.layers.Conv2D(16, 3)
        self.conv2 = tf.keras.layers.Conv2D(16, 3)
        self.conv3 = tf.keras.layers.Conv2D(3, 3)

        self.simulator = Simulator(type)

        pass

    def call(self, inputs):
        # TODO: Write forward-pass logic
        
        # linear corrector layer
        simulator_difference = inputs - self.simulator(inputs)
        linear_output = inputs + tf.matmul(self.linear_corrector, simulator_difference)

        # convolutional layers
        conv1_output = self.conv1(linear_output)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)

        return conv3_output
    
    def loss(self, probs, labels):
        # TODO: Write loss function
        pass
    
