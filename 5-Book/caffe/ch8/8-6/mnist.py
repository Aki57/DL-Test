import tensorflow as tf
import numpy as np
import os

class MNist:
    def __init__(self):
        self.X = tf.placeholder("float", [None, 28, 28, 1])
        self.Y = tf.placeholder("float", [None, 10])
        predictY = self.model(self.X)
        self.cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictY, self.Y))
        self.accuracy_op = tf.contrib.metrics.accuracy(tf.argmax(predictY, 1), tf.argmax(self.Y,
            1))
        self.batch_size = 128
        self.test_size = 1024

    def model(self, X):
        w = tf.Variable(tf.random_normal([5, 5, 1, 20], stddev=0.01))
        w2 = tf.Variable(tf.random_normal([5, 5, 20, 50], stddev=0.01))
        w3 = tf.Variable(tf.random_normal([4 * 4 * 50, 500], stddev=0.01))
        w4 = tf.Variable(tf.random_normal([500, 10], stddev=0.01))

        l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides = [1, 1, 1, 1], padding='VALID'))
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides = [1, 1, 1, 1], padding='VALID'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        l2 = tf.reshape(l2, [-1, 4 * 4 * 50])
        l3a = tf.nn.relu(tf.matmul(l2,w3))

        y = tf.matmul(l3a, w4)
        self.params = [w, w2, w3, w4]
        return y

    def train(self, trainX, trainY, testX, testY):
        train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(self.cost_op)
        
        with tf.Session() as sess:
            ckpt_dir = "./ckpt_dir"
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            saver.save(sess, ckpt_dir + "/mnist_init.ckpt",global_step = 0)

            for i in range(10):
                training_batch = zip(range(0, len(trainX), self.batch_size), range(self.batch_size, len(trainX) + 1, self.batch_size))
                for start, end in training_batch:
                    sess.run(train_op, feed_dict={self.X: trainX[start:end], self.Y:trainY[start:end]})

                test_indices = np.arange(len(testX))
                np.random.shuffle(test_indices)
                test_indices = test_indices[0:self.test_size]
                
                print(i, sess.run(self.accuracy_op, \
                    feed_dict={self.X:testX[test_indices], \
                        self.Y:testY[test_indices]}))
            saver.save(sess, ckpt_dir + "/mnist.ckpt", global_step=10000)
