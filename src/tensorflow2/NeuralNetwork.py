from config import CFG
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import eval as k_eval
from src.tensorflow2.network import AlphaGoNet, alpha_loss


# a cosa serve sta roba
@tf.function
def train_step(model, optimizer, input_data, z, pi):
  with tf.GradientTape() as tape:
    z_pred, pi_pred = model(input_data)
    loss, loss_policy, loss_value = alpha_loss(z, z_pred, pi, pi_pred)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss, loss_policy, loss_value


class NeuralNetwork(object):
    def __init__(self, game, model_name=None):
        self.game = game
        self.age = 0
        self.optimizer = SGD
        self.session_initialize()
        # if model_name:
        #     self.load_model(model_name)

    def session_initialize(self):
        self.net = AlphaGoNet(self.game)
        self.net.compile(optimizer='SGD',
                         loss=alpha_loss,
                         metrics=['accuracy'])

    def eval(self, state):
        board = state.board.board_repr(state.player_color)
        p, v = self.net(tf.dtypes.cast(board, tf.float32))
        return k_eval(p[0]), k_eval(v[0][0])

    def train(self, input_data, output_data_pi, output_data_z):

        for epoch in range(CFG.epochs):

            loss_policy_mean, loss_value_mean, loss_mean, j = 0, 0, 0, 0

            for i, pi, z in zip(input_data, output_data_pi, output_data_z):

                history = self.net.fit(i, (pi, z), epochs=CFG.epochs)
                print(history)
                # loss, _, _ = train_step(self.net, self.optimizer, i, z, pi)

                # loss_mean = (loss_mean * j + loss) / (j + 1)

                j += 1

            print(f"Epoch: {epoch + 1} - loss mean = {loss_mean}\n"
                  f"loss policy mean = {loss_policy_mean}\n"
                  f"loss value mean = {loss_value_mean}")

    # def save_model(self, filename):
    #     print("Saving model:", filename, "at", CFG.model_directory)
    #     self.saver.save(self.sess, filename)

    # def load_model(self, filename):
    #     print("Loading model:", filename, "from", CFG.model_directory)
    #     self.saver.restore(self.sess, filename)
