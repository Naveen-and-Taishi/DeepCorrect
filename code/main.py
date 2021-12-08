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

def run_d1(referee, d1):
    # TODO: train referee
    for epoch in range(100):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in d1:
            print("Batch " + str(batch_counter))
            batch_counter += 1
            with tf.GradientTape() as tape:
                probs = referee.call(batch['image'])
                loss = referee.loss(probs, batch['labels'][:, 0])
                print("loss: " + str(loss)) 
            gradients = tape.gradient(loss, referee.trainable_variables)
            referee.Adam.apply_gradients(zip(gradients, referee.trainable_variables))


def run_d2(corrector, referee, d2):
    # TODO: train corrector and test referee
    for epoch in range(100):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in d2:
            print("Batch " + str(batch_counter))
            batch_counter += 1
            with tf.GradientTape() as tape:
                corrected_images = corrector(batch['image'])
                simulated_corrected_images = corrector.simulator.simulate_image(corrected_images)
            # we don't want to alter the referee at this point, we it is outside the gradient tape
            loss = referee.loss(simulated_corrected_images, batch['labels'])
            gradients = tape.gradient(loss, referee.trainable_variables)
            referee.Adam.apply_gradients(zip(gradients, referee.trainable_variables))
            

def run_d3(corrector, d3):
    # TODO: test corrector

    # In the paper it says the corrector's results were compared with those of the linear corrector and with normal images

    pass

def main():

    # 100 as batch_size for now, change later
    batch_size = 20

    corrector_deuteranope = Corrector(batch_size, 'D')
    corrector_protanope = Corrector(batch_size, 'P')
    corrector_tritanope = Corrector(batch_size, 'T')

    referee = Referee(batch_size)

    d1, d2, d3 = load_data(batch_size)

    # # testing that data loaded correctly
    # for batch in d1:
    #     print(batch['image'])
    #     print(batch['labels'][:, 0])
    #     break

    # TODO: Train and test Corrector and Referee models 

    print("STARTING D1")
    run_d1(referee, d1)
    print("STARTING D2")
    run_d2(corrector_deuteranope, referee, d2)
    run_d2(corrector_protanope, referee, d2)
    run_d2(corrector_tritanope, referee, d2)
    print("STARTING D3")
    accuracy_deuteranope = run_d3(corrector_deuteranope, d3)
    accuracy_protanope = run_d3(corrector_protanope, d3)
    accuracy_tritanope = run_d3(corrector_tritanope, d3)

    print("ACCURACY DEUTERANOPE: " + accuracy_deuteranope)
    print("ACCURACY PROTANOPE: " + accuracy_protanope)
    print("ACCURACY TRITANOPE: " + accuracy_tritanope)

    # TODO: we can now use the trained corrector models to visualize some results here

    return

if __name__ == "__main__":
    main()