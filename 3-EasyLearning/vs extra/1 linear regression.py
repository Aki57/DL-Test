import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow as tf

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
    rng = np.random

    # Parameters
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 50

    #Training Data
    train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                          7.042,10.791,5.313,7.997,5.654,9.27,3.1])
    train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                          2.827,3.465,1.65,2.904,2.42,2.94,1.3])
    n_samples = train_X.shape[0]

    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # Set model weights
    W = tf.Variable(rng.randn(), name = "weight")
    b = tf.Variable(rng.randn(), name = "bias")

    # Construct a linear model
    pred = tf.add(tf.multiply(X,W), b)

    # Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2)/(2*n_samples))
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x,y) in zip(train_X,train_Y):
                sess.run(optimizer, feed_dict={X:x,Y:y})

            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

        print("Optimization Finished!")
        training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '/n')
        
        # Graphic Display
        plt.plot(train_X, train_Y, 'ro', label='Original Data')
        plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted Line')
        plt.legend()
        plt.show()

    exit(0)


if __name__ == "__main__":
    tf.app.run()
