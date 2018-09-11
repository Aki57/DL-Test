import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

###################################################################
# Variables                                                       #
# When launching project or scripts from Visual Studio,           #
# input_dir and output_dir are passed as arguments automatically. #
# Users could set them from the project setting page.             #
###################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("input_dir", ".", "Input directory where training dataset and meta data are saved")
tf.app.flags.DEFINE_string("output_dir", ".", "Output directory where output such as logs are saved.")
tf.app.flags.DEFINE_string("log_dir", ".", "Model directory where final model files are saved.")

def main(_):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

    # In this example, we limit mnist data
    Xtr, Ytr = mnist.train.next_batch(5000) # 5000 for training
    Xte, Yte = mnist.train.next_batch(200) # 200 for testing

    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.argmin(distance, 0)
    accuracy = 0.

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),
                  "True Class:", np.argmax(Yte[i]))
            # Calculate accuracy
            if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
                accuracy += 1./ len(Xte)

        print("Done!")
        print("Accuracy:", accuracy)
        
    exit(0)


if __name__ == "__main__":
    tf.app.run()
