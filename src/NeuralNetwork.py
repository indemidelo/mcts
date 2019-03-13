import tensorflow as tf
from src.singleton import Singleton
from src.network import AlphaGo19Net


class NeuralNetwork(metaclass=Singleton):
    def __init__(self):
        self.beta = 1e-4
        self.n_res_blocks = 1
        self.learning_rate = 0.001
        self.initialize()

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

    def eval(self, board):
        p, v = self.sess.run([self.pred_policy, self.pred_value],
            feed_dict={self.inputs: board})
        return p[0], v[0][0]

    def train(self, input_data, pi_output, z_output, n_epochs):
        feed_dict = {
            self.inputs: input_data,
            self.pi: pi_output,
            self.z: z_output}
        for epoch in range(n_epochs):
            _, c, acc_policy, acc_value = self.sess.run(
                [self.optimizer, self.loss,
                 self.acc_policy, self.acc_value],
                feed_dict=feed_dict)
            # if (epoch + 1) % 25:
            print(f"Epoch: {epoch + 1} - cost= {c}\n"
                  f"accuracy policy: {acc_policy} - "
                  f"accuracy value: {acc_value}")

    def save(self, iter):
        print(f'Saving at iter {iter}')
