from config import CFG
import torch
import torch.optim as optim
from src.pytorch.network import AlphaGoNet, AlphaLoss

device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')


class NeuralNetwork(object):

    def __init__(self, game, model_name=None):
        self.game = game
        self.age = 0
        self.session_initialize()
        if model_name:
            self.load_model(model_name)

    def session_initialize(self):
        self.net = AlphaGoNet(self.game)

        if CFG.gpu_train:
            self.net.cuda()

        self.criterion = AlphaLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum)

    def eval(self, state):
        board = state.board.board_repr(state.player_color)
        board = to_tensor(board)
        p, v = self.net(board)
        p, v = to_array(p, v)
        return p[0], v[0][0]

    def train(self, input_data, output_data_pi, output_data_z):

        for epoch in range(CFG.epochs):

            loss_policy_mean, loss_value_mean, loss_mean, j = .0, .0, .0, 0

            for i, pi, z in zip(input_data, output_data_pi, output_data_z):
                i_t, pi_t, z_t = to_tensor(i, pi, z)

                self.optimizer.zero_grad()
                pred_pi, pred_z = self.net(i_t)

                loss, loss_value, loss_policy = self.criterion(
                    z_t, pred_z, pi_t, pred_pi)

                loss.backward()
                self.optimizer.step()

                # print statistics
                # loss_policy_mean = (loss_policy_mean * j + loss_policy.detach()) / (j + 1)
                # loss_value_mean = (loss_value_mean * j + loss_value.detach()) / (j + 1)
                loss_mean = (loss_mean * j + loss.detach()) / (j + 1)

                j += 1

            print(f"Epoch: {epoch + 1} - loss mean = {loss_mean}")  # \n")
            # f"loss policy mean = {loss_policy_mean}\n"
            # f"loss value mean = {loss_value_mean}")

    def save_model(self, filename):
        print("Saving model:", filename, "at", CFG.model_directory)
        # self.saver.save(self.sess, filename)

    def load_model(self, filename):
        print("Loading model:", filename, "from", CFG.model_directory)
        # self.saver.restore(self.sess, filename)


def to_tensor(*args):
    """
    Convert a list of numpy arrays in pytorch tensors
    :param args: list of arrays
    :return: list of tensors or single tensor
    """
    tensors = list()
    for array in args:

        t = torch.from_numpy(array)
        t = t.to(dtype=torch.float)
        if CFG.gpu_train:
            t = t.to(device_gpu)

        tensors.append(t)

    if len(tensors) == 1:
        return tensors[0]

    return tensors


def to_array(*args):
    """
    Convert a list of pytorch tensors in numpy arrays
    :param args: list of tensors
    :return: list of arrays or single array
    """
    arrays = list()
    for tensor in args:
        t = tensor.to(device_cpu) if CFG.gpu_train else tensor
        arrays.append(t.detach().numpy())
    if len(arrays) == 1:
        return arrays[0]
    return arrays
