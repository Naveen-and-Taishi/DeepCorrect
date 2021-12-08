import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math

from corrector import Corrector
from referee import Referee

def load_data(batch_size):
    
    # # Create 3 disjoint subsets of the PASCAL dataset
    # d1 = tfds.load('voc', split='train', batch_size=batch_size)
    # d2 = tfds.load('voc', split='test', batch_size=batch_size)
    # d3 = tfds.load('voc', split='validation', batch_size=batch_size)

    # assert isinstance(d1, tf.data.Dataset)
    # assert isinstance(d2, tf.data.Dataset)
    # assert isinstance(d3, tf.data.Dataset)

    # return d1, d2, d3

    (d12_data, d12_labels), (d3_data, d3_labels) = tf.keras.datasets.cifar100.load_data()

    d1 = (d12_data[:25000] / 255, d12_labels[:25000], 100)
    d2 = (d12_data[25000:] / 255, d12_labels[25000:], 100)
    d3 = (d3_data / 255, d3_labels, 100)

    return d1, d2, d3

def run_d1(referee, d1):
    # TODO: train referee
    for epoch in range(100):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in range(0, len(d1[0]), referee.batch_size):
            print("Batch " + str(batch_counter))
            batch_counter += 1
            with tf.GradientTape() as tape:
                logits = referee.call(d1[0][batch : batch + referee.batch_size])
                loss = referee.loss(logits, d1[1][batch : batch + referee.batch_size])
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
            loss = referee.loss(simulated_corrected_images, batch['labels'][:, 0])
            gradients = tape.gradient(loss, referee.trainable_variables)
            referee.Adam.apply_gradients(zip(gradients, referee.trainable_variables))
            

def run_d3(corrector, referee, d3):
    # TODO: test our corrected images vs uncorrected images

    total_acc_corrected = 0
    total_acc_uncorrected = 0
    batch_counter = 0

    for batch in d3:
        print("Batch " + str(batch_counter))
        batch_counter += 1
        corrected_images = corrector(batch['image'])
        uncorrected_images = batch['image']
        corrected_pred = referee.call(corrected_images)
        uncorrected_pred = referee.call(uncorrected_images)
        total_acc_corrected = tf.equal(tf.argmax(corrected_pred, 1), tf.argmax(batch['labels'][:, 0], 1))
        total_acc_uncorrected = tf.equal(tf.argmax(uncorrected_pred, 1), tf.argmax(batch['labels'][:, 0], 1))

    return total_acc_corrected / batch_counter, total_acc_uncorrected / batch_counter

def main():

    # 100 as batch_size for now, change later
    batch_size = 100

    corrector_deuteranope = Corrector(batch_size, 'D')
    corrector_protanope = Corrector(batch_size, 'P')
    corrector_tritanope = Corrector(batch_size, 'T')

    referee = Referee(batch_size)

    d1, d2, d3 = load_data(batch_size)

    # testing that data loaded correctly

    # TODO: Train and test Corrector and Referee models 

    print("STARTING D1")
    run_d1(referee, d1)
    # print("STARTING D2")
    # run_d2(corrector_deuteranope, referee, d2)
    # run_d2(corrector_protanope, referee, d2)
    # run_d2(corrector_tritanope, referee, d2)
    # print("STARTING D3")
    # accuracy_deuteranope = run_d3(corrector_deuteranope, referee, d3)
    # accuracy_protanope = run_d3(corrector_protanope, referee, d3)
    # accuracy_tritanope = run_d3(corrector_tritanope, referee, d3)

    # print("ACCURACY DEUTERANOPE: " + accuracy_deuteranope)
    # print("ACCURACY PROTANOPE: " + accuracy_protanope)
    # print("ACCURACY TRITANOPE: " + accuracy_tritanope)

    # TODO: we can now use the trained corrector models to visualize some results here

    return

if __name__ == "__main__":
    main()