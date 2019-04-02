import tensorflow as tf
from src.config import CFG
from src.network import AlphaGo19Net


class NeuralNetwork(object):
    def __init__(self, model_name=None):
        self.age = 0
        self.session_initialize()
        if model_name:
            self.load_model(model_name)

    def session_initialize(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, 6, 7, 3])
            self.pi = tf.placeholder(tf.float32, [None, 7])
            self.z = tf.placeholder(tf.float32, [None, 1])
            self.pred_policy, self.pred_value, self.loss, self.optimizer, \
            self.loss_policy, self.loss_value = AlphaGo19Net(
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
            loss_policy_mean, loss_value_mean, loss_mean, j = 0, 0, 0, 0
            for i, pi, z in zip(input_data, output_data_pi, output_data_z):
                feed_dict = {self.inputs: i, self.pi: pi, self.z: z}
                _, c, loss_policy, loss_value = self.sess.run(
                    [self.optimizer, self.loss, self.loss_policy,
                     self.loss_value], feed_dict=feed_dict)
                loss_policy_mean = (loss_policy_mean * j + loss_policy) / (j + 1)
                loss_value_mean = (loss_value_mean * j + loss_value) / (j + 1)
                loss_mean = (loss_mean * j + c) / (j + 1)
                j += 1
            print(f"Epoch: {epoch + 1} - loss mean = {loss_mean}\n"
                  f"loss policy mean = {loss_policy_mean}\n"
                  f"loss value mean = {loss_value_mean}")

    def save_model(self, filename):
        print("Saving model:", filename, "at", CFG.model_directory)
        self.saver.save(self.sess, filename)

    def load_model(self, filename):
        print("Loading model:", filename, "from", CFG.model_directory)
        self.saver.restore(self.sess, filename)
