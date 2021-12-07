import os
import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.keras.applications.vgg16 import VGG16

class Referee(tf.keras.Model):
    def __init__(self) -> None:
        super(Referee, self).__init__()

        # TODO: Initialize all hyperparameters

        
        
        pass

    def call(self, inputs):
        # TODO: Write forward-pass logic
        pass
    
    def loss(self, probs, labels):
        # TODO: Write loss function
        pass
    
