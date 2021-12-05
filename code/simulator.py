import os
import tensorflow as tf
import numpy as np
import random
import math

class Simulator():
    def __init__(self, type) -> None:
        if type == 'D':
            # deuteranope
            self.color_matrix = tf.convert_to_tensor([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])
        elif type == 'P':
            # protanope
            self.color_matrix = tf.convert_to_tensor([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
        elif type == 'T':
            # tritanope
            self.color_matrix = tf.convert_to_tensor([[1, 0, 0], [0, 1, 0], [-0.395913, 0.801109, 0]])
        else:
            raise("ERROR: invalid type passed into Simulator class (only accepts 'D', 'P', or 'T')")

        self.rgb2lms = tf.convert_to_tensor([[17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]])

    def simulate_image(self, image):
        # passes an image through the color-blindness simulator
        
        inverted_rgb2lms = tf.matrix_inverse(self.rgb2lms)

        product1 = tf.tensordot(inverted_rgb2lms, self.color_matrix)
        product2 = tf.tensordor(product1, self.rgb2lms)
        simulated_image = tf.tensordot(product2, image)

        return simulated_image