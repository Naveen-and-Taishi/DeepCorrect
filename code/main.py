import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math

from corrector import Corrector
from referee import Referee
from simulator import Simulator

def load_data(batch_size):
    
    # Create 3 disjoint subsets of the PASCAL dataset
    d1 = tfds.load('voc', split='train', batch_size=batch_size)
    d2 = tfds.load('voc', split='test', batch_size=batch_size)
    d3 = tfds.load('voc', split='validation', batch_size=batch_size)

    assert isinstance(d1, tf.data.Dataset)
    assert isinstance(d2, tf.data.Dataset)
    assert isinstance(d3, tf.data.Dataset)

    return d1, d2, d3

def main():

    corrector = Corrector()
    referee = Referee()

    # 100 as batch_size for now, change later
    d1, d2, d3 = load_data(100)


    # TODO: Train Corrector and Refereee models 

    # TODO: Test

    # testing that data loaded correctly
    for batch in d1:
        print(batch)
        break

    pass

if __name__ == "__main__":
    main()