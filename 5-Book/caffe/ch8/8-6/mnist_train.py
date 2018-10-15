import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
from mnist import MNist

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)


m = MNist()
m.train(trX, trY, teX, teY)
