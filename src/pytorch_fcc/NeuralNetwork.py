from config import CFG
from copy import deepcopy
import torch
import numpy as np

device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')


class NeuralNetwork(object):

    def __init__(self, game, model_name=None):
        in_ch, h, w = self.game.input_shape()
        game_dim = in_ch * h * w
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG.learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(game_dim, game_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(game_dim * 2, game_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(game_dim * 4, game_dim * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(game_dim * 4, 1))
        self.game = game
        self.age = 0
        if CFG.gpu_train:
            self.model.cuda()
        if model_name:
            self.load_model(model_name)

    def eval(self, state):
        board = state.board.simpler_board_repr(state.player_color)
        in_ch, h, w = self.game.input_shape()
        p = [np.zeros((in_ch + 1, h, w)) for _ in range(self.game.policy_shape())]

        for move in state.board.list_available_moves():
            board_clone = deepcopy(state.board)
            board_clone.play_(state.player_color, move)
            board_clone = state.board_clone.simpler_board_repr(-state.player_color)
            board_clone = to_tensor(board_clone)
            p[move] = self.model(board_clone)

        board = to_tensor(board)
        v = self.model(board)
        return p, v

    def train(self, input_data, output_data_v):

        for epoch in range(CFG.epochs):

            loss_mean, j = 0.0, 0

            for s, v in zip(input_data, output_data_v):
                state_t, v_target = to_tensor(s, v)

                self.optimizer.zero_grad()
                v_output = self.model(state_t)
                loss = self.loss_fn(v_output, v_target)
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
