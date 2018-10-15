import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf
from tensorflow.contrib import rnn
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
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Parameters
    learning_rate = 0.001
    training_epochs = 100000
    batch_size = 128
    display_step = 10

    # Network Parameters
    n_input = 28 # MNIST data input (img shape: 28*28)
    n_steps = 28 # timesteps
    n_hidden = 128 # hidden layer num of features
    n_classes = 10 # MNIST total classes (0-9 digits)

    # tf Graph Input
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Create model
    def RNN(x, weights, biases):
        # Rrepare data shape to match 'rnn' function requirements
        # Current data input shape: (batch_size, n_steps, n_nput)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_imput)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, n_steps, 1)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    # Construct model
    pred = RNN(x, weights, biases)
    
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_epochs:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                # Calcualte batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " + 
                      "{:.5f}".format(acc))
            
            step += 1

        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        test_len = 128
        test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
        test_label = mnist.test.labels[:test_len]
        print("Testing Accuracy", sess.run(accuracy, feed_dict={x: test_data,y: test_label}))

    exit(0)
    
if __name__ == "__main__":
    tf.app.run()
