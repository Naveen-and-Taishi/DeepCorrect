import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random
import math

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('tkagg')

from corrector import Corrector
from referee import Referee

def compute_accuracy(logits, labels):
    correct_predictions = tf.equal(tf.argmax(tf.squeeze(logits), 1), tf.squeeze(labels))
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def load_data():
    (d12_data, d12_labels), (d3_data, d3_labels) = tf.keras.datasets.cifar100.load_data()

    d1 = (d12_data[:25000] / 255, d12_labels[:25000], 100)
    d2 = (d12_data[25000:] / 255, d12_labels[25000:], 100)
    d3 = (d3_data / 255, d3_labels, 100)

    return d1, d2, d3

def run_d1(referee, d1):
    # TODO: train referee
    for epoch in range(1):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in range(1):#0, len(d1[0]), referee.batch_size):
            print("Batch " + str(batch_counter))
            batch_counter += 1
            labels = d1[1][batch : batch + referee.batch_size]
            with tf.GradientTape() as tape:
                logits = referee.call(d1[0][batch : batch + referee.batch_size])
                loss = referee.loss(logits, tf.squeeze(labels))
                print("loss: " + str(loss)) 
            accuracy = compute_accuracy(logits, labels)
            print("ACCURACY: " + str(accuracy))
            gradients = tape.gradient(loss, referee.trainable_variables)
            referee.Adam.apply_gradients(zip(gradients, referee.trainable_variables))


def run_d2(corrector, referee, d2):
    # TODO: train corrector and test referee
    for epoch in range(2):
        print("EPOCH " + str(epoch))
        batch_counter = 0
        for batch in range(0, len(d2[0]), referee.batch_size):
            print("Batch " + str(batch_counter))
            batch_counter += 1
            labels = d2[1][batch : batch + referee.batch_size]
            with tf.GradientTape() as tape:
                corrected_images = corrector(d2[0][batch : batch + referee.batch_size])
                simulated_corrected_images = tf.map_fn(corrector.simulator.simulate_image, corrected_images)
                logits = referee.call(simulated_corrected_images)
                loss = referee.loss(logits, labels)
                print("loss: " + str(loss)) 
                accuracy = compute_accuracy(logits, labels)
                print("ACCURACY: " + str(accuracy))
            gradients = tape.gradient(loss, corrector.trainable_variables)
            corrector.Adam.apply_gradients(zip(gradients, corrector.trainable_variables))
            

def run_d3(corrector, referee, d3):
    # TODO: test our corrected images vs uncorrected images

    total_acc_corrected = 0
    total_acc_uncorrected = 0
    batch_counter = 0

    for batch in range(0, len(d3[0]), referee.batch_size):
        print("Batch " + str(batch_counter))
        batch_counter += 1
        corrected_images = corrector(d3[0][batch : batch + referee.batch_size])
        uncorrected_images = d3[0][batch : batch + referee.batch_size]
        labels = d3[1][batch : batch + referee.batch_size]
        corrected_pred = referee.call(corrected_images)
        uncorrected_pred = referee.call(uncorrected_images)
        acc_corrected = compute_accuracy(corrected_pred, labels)
        acc_uncorrected = compute_accuracy(uncorrected_pred, labels)
        print("Accuracy with correction: " + str(acc_corrected))
        print("Accuracy without correction: " + str(acc_uncorrected))
        total_acc_corrected += acc_corrected
        total_acc_uncorrected += acc_uncorrected

    return total_acc_corrected / batch_counter, total_acc_uncorrected / batch_counter

def main():

    # 100 as batch_size for now, change later
    batch_size = 500

    corrector_deuteranope = Corrector(batch_size, 'D')
    corrector_protanope = Corrector(batch_size, 'P')
    corrector_tritanope = Corrector(batch_size, 'T')

    referee = Referee(batch_size)

    d1, d2, d3 = load_data()

    # testing that data loaded correctly

    # TODO: Train and test Corrector and Referee models 

    print("STARTING D1")

    # save the model so we don't have to train it again
    #run_d1(referee, d1)
    #referee.save_weights('../models/referee.tf')
    
    # load weights from saved model
    referee.load_weights('../models/referee.tf')
    #run for one batch to initialize params
    run_d1(referee, d1)


    print("STARTING D2")
    run_d2(corrector_deuteranope, referee, d2)
    corrector_deuteranope.save_weights('../models/corrector_deuteranope.tf')
    run_d2(corrector_protanope, referee, d2)
    corrector_protanope.save_weights('../models/corrector_protanope.tf')
    
    #corrector_protanope.load_weights('../models/corrector_protanope.tf')

    run_d2(corrector_tritanope, referee, d2)
    corrector_tritanope.save_weights('../models/corrector_tritanope.tf')
    
    print("STARTING D3")
    accuracy_deuteranope_corrected, accuracy_deuteranope_uncorrected = run_d3(corrector_deuteranope, referee, d3)
    accuracy_protanope_corrected, accuracy_protanope_uncorrected = run_d3(corrector_protanope, referee, d3)
    accuracy_tritanope_corrected, accuracy_tritanope_uncorrected = run_d3(corrector_tritanope, referee, d3)

    print("ACCURACY DEUTERANOPE CORRECTED: " + str(accuracy_deuteranope_corrected))
    print("ACCURACY DEUTERANOPE UNCORRECTED: " + str(accuracy_deuteranope_uncorrected))
    print("ACCURACY PROTANOPE CORRECTED: " + str(accuracy_protanope_corrected))
    print("ACCURACY PROTANOPE UNCORRECTED: " + str(accuracy_protanope_uncorrected))
    print("ACCURACY TRITANOPE CORRECTED: " + str(accuracy_tritanope_corrected))
    print("ACCURACY TRITANOPE UNCORRECTED: " + str(accuracy_tritanope_uncorrected))

    # TODO: we can now use the trained corrector models to visualize some results here
    nc = 7
    nr = 10

    fig = plt.figure()

    image = d3[0][0]
    image_idx = 0

    for i in range(70):
        ax = fig.add_subplot(nr, nc, i+1)

        if (i % 7 == 0):
            image = d3[0][i]
            image_idx = i
            ax.imshow(image, cmap="Greys")
        elif (i % 7 == 1):
            ax.imshow(corrector_deuteranope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 2):
            ax.imshow(corrector_protanope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 3):
            ax.imshow(corrector_tritanope.simulator.simulate_image(tf.cast(image, dtype=tf.float32)))
        elif (i % 7 == 4):
            ax.imshow(corrector_deuteranope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])
        elif (i % 7 == 5):
            ax.imshow(corrector_protanope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])
        else:
            ax.imshow(corrector_tritanope.call(tf.cast(d3[0][image_idx : image_idx + 100], dtype=tf.float32))[0])

    plt.show()

    return

if __name__ == "__main__":
    main()