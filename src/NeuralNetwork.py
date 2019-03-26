import tensorflow as tf
from src.config import CFG
from src.singleton import Singleton
from src.network import AlphaGo19Net


class NeuralNetwork(metaclass=Singleton):
    def __init__(self):
        self.inputs = tf.placeholder(tf.float32, [None, 6, 7, 3], name='InputData')
        self.pi = tf.placeholder(tf.float32, [None, 7], name='pi')
        self.z = tf.placeholder(tf.float32, [None, 1], name='z')
        self.pred_policy, self.pred_value, self.loss, self.optimizer, \
        self.acc_policy, self.acc_value = AlphaGo19Net(
            self.inputs, self.pi, self.z)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def eval(self, state):
        board = state.board.board_as_tensor(state.player_color)
        p, v = self.sess.run([self.pred_policy, self.pred_value],
                             feed_dict={self.inputs: board})
        return p[0], v[0][0]

    def train(self, input_data, output_data_pi, output_data_z):
        for epoch in range(CFG.epochs):
            acc_policy_mean, acc_value_mean, loss_mean, j = 0, 0, 0, 0
            for i, pi, z in zip(input_data, output_data_pi, output_data_z):
                feed_dict = {self.inputs: i, self.pi: pi, self.z: z}
                _, c, acc_policy, acc_value = self.sess.run(
                    [self.optimizer, self.loss, self.acc_policy,
                     self.acc_value], feed_dict=feed_dict)
                acc_policy_mean = (acc_policy_mean * j + acc_policy) / (j + 1)
                acc_value_mean = (acc_value_mean * j + acc_value) / (j + 1)
                loss_mean = (loss_mean * j + c) / (j + 1)
                j += 1
                # print('z:', z.reshape([1, -1]))
            print(f"Epoch: {epoch + 1} - cost mean= {loss_mean}\n"
                  f"accuracy policy mean= {acc_policy_mean}\n"
                  f"accuracy value mean= {acc_value_mean}")

    def save(self, iter):
        self.saver.save(self.sess, f'{CFG.model_directory}/my_little_model_iter_{iter}')
