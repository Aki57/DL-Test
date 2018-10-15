import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

from mnist import MNist
from opt_line import OptLine

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
teX = teX.reshape(-1, 28, 28, 1)

model = MNist()
ol = OptLine(model.params)
res = ol.gen_loss_line('./ckpt_dir/mnist_init.ckpt-0','./ckpt_dir/mnist.ckpt-10000', model, teX, teY)
for i in res:
	print i
