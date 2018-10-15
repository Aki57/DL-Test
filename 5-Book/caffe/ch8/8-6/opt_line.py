import tensorflow as tf
import numpy as np
import os

class OptLine:
    def __init__(self, param_list):
        self.param_list = param_list
        self.batch_size = 128

    def gen_loss_line(self, init_ckpt, opt_ckpt, model, test_data, test_label):
        res = []
        batchNum = len(test_label) / self.batch_size + (0 if len(test_label) % 128 ==0 else 1)
        if batchNum == 0:
            return
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, init_ckpt)
            init_val = []
            for i, param in enumerate(self.param_list):
                init_val.append(sess.run(param))
            saver.restore(sess, opt_ckpt)
            opt_val = []
            for i, param in enumerate(self.param_list):
                opt_val.append(sess.run(param))
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for rate in range(0, 101):
                for i, param in enumerate(self.param_list):
                    temp_val = (init_val[i] * (100 - rate) + opt_val[i] * rate) / 100
                    assign_op = param.assign(temp_val)
                    sess.run(assign_op)
                acc = 0
                count = 0
                test_batch = zip(range(0, len(test_data), self.batch_size), range(self.batch_size, len(test_data) + 1, self.batch_size))
                for start, end in test_batch:
                    loss = sess.run(model.cost_op, feed_dict={model.X:test_data[start:end], model.Y: test_label[start:end]})
                    acc += loss * (end - start)
                    count += end - start
                res.append((rate, acc / count))
        return res
