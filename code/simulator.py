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
        
        inverted_rgb2lms = tf.linalg.inv(self.rgb2lms)

        product1 = tf.matmul(inverted_rgb2lms, self.color_matrix)
        product2 = tf.matmul(product1, self.rgb2lms)

        original_image_shape = image.shape
        
        simulated_image = tf.transpose(tf.matmul(product2, tf.reshape(tf.transpose(image, perm=[2, 0, 1]), (image.shape[2], image.shape[0] * image.shape[1]))), perm=[1, 0])

        return tf.reshape(simulated_image, original_image_shape)