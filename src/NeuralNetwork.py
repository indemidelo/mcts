import tensorflow as tf
from src.singleton import Singleton
from src.network import AlphaGo19Net


class NeuralNetwork(metaclass=Singleton):
    def __init__(self):
        self.beta = 0.5
        self.n_res_blocks = 19
        self.learning_rate = 0.001
        pass

    def initialize(self):
        self.inputs = tf.placeholder(tf.float32, [None, 6, 7, 3], name='InputData')
        self.pi = tf.placeholder(tf.float32, [None, 7], name='pi')
        self.z = tf.placeholder(tf.float32, [None, 1], name='z')
        self.pred_policy, self.pred_value, self.loss, self.optimizer, \
        self.acc_policy, self.acc_value = AlphaGo19Net(
            self.inputs, self.pi, self.z, self.beta,
            self.n_res_blocks, self.learning_rate)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def eval(self, state):
        p = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.35, 4: 0.05, 5: 0.0, 6: 0.1}
        v = 0.6
        return p, v

    def train(self, *args):
        feed_dict = {}
        _, c, acc_policy, acc_value = self.sess.run(
            [self.optimizer, self.loss,
             self.acc_policy, self.acc_value],
            feed_dict=feed_dict)

    def save(self, iter):
        print(f'Saving at iter {iter}')
