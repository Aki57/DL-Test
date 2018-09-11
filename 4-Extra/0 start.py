import numpy as np
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
    # tf graph input
    a = tf.placeholder(tf.int16)
    b = tf.placeholder(tf.int16)

    add = tf.add(a,b)
    mul = tf.multiply(a,b)
    
    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])

    product = tf.matmul(matrix1, matrix2)

    with tf.Session() as sess:
        print("Addition with constants: %i" % sess.run(add, feed_dict = {a: 2,b: 3}))
        print("Multiplication with constants: %i" % sess.run(mul, feed_dict = {a: 2,b: 3}))
        
        result = sess.run(product)
        print(result)

    exit(0)


if __name__ == "__main__":
    tf.app.run()
