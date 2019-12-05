from config import CFG
from copy import deepcopy
import torch
import numpy as np

device_gpu = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')

def softmax(x):
    """
    return the softmax for x
    :param x: scalar or array
    :return:
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class NeuralNetwork(object):

    def __init__(self, game, model_name=None):
        game_dim = np.prod(game.input_shape())
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(game_dim, game_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(game_dim * 2, game_dim * 4),
            torch.nn.ReLU(),
            # torch.nn.Linear(game_dim * 4, game_dim * 4),
            # torch.nn.ReLU(),
            torch.nn.Linear(game_dim * 4, 1))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CFG.learning_rate)
        self.game = game
        self.age = 0
        if CFG.gpu_train:
            self.model = self.model.double().cuda(device_gpu)
        if model_name:
            self.load_model(model_name)

    def eval(self, state):
        boards_to_eval = state.board.board_repr(state.player_color)
        available_moves = state.board.list_available_moves()
        p = {j: 0.0 for j in range(self.game.policy_shape())}

        for move in available_moves:
            board_clone = deepcopy(state.board)
            board_clone.play_(state.player_color, move)
            board_clone = board_clone.board_repr(-state.player_color)
            boards_to_eval = np.append(boards_to_eval, board_clone, axis=0)

        # output = self.model(to_tensor(np.array(boards_to_eval)))
        # input = torch.tensor(boards_to_eval).double()
        # input = input.view(input.shape[0], -1)
        # input = input.to(device_gpu)
        input = torch.tensor(boards_to_eval).double().to(device_gpu)
        # output = self.flatten(self.model(input))
        with torch.no_grad():
            self.model.eval()
            output = self.model(input)
            output = output.view(output.shape[0])

        if CFG.gpu_train:
            output = output.detach().cpu().numpy()
        v = output[0]
        p_lam = dict(zip(available_moves, list(output[1:])))
        p = softmax(-np.asarray(list({**p, **p_lam}.values())))
        return p, v

    def train(self, input_data, _, output_data_v):

        for epoch in range(CFG.epochs):

            loss_mean, j = 0.0, 0

            for s, v in zip(input_data, output_data_v):
                # state_t, v_target = to_tensor(s, v)

                self.optimizer.zero_grad()
                state_t = torch.tensor(s).double().to(device_gpu)
                v_target = torch.tensor(v).double().to(device_gpu)
                v_output = self.model(state_t)
                v_output_flat = v_output.view(v_output.shape[0])
                loss = self.loss_fn(v_output_flat, v_target)
                loss.backward()
                self.optimizer.step()

                # print statistics
                # loss_policy_mean = (loss_policy_mean * j + loss_policy.detach()) / (j + 1)
                # loss_value_mean = (loss_value_mean * j + loss_value.detach()) / (j + 1)
                loss_mean = (loss_mean * j + loss.detach()) / (j + 1)

                j += 1

            if (epoch + 1) % 5 == 0:
                print(f"Epoch: {epoch + 1} - loss mean = {loss_mean}")  # \n")
                # f"loss policy mean = {loss_policy_mean}\n"
                # f"loss value mean = {loss_value_mean}")

    def save_model(self, filename):
        print("Saving model:", filename, "at", CFG.model_directory)
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        print("Loading model:", filename, "from", CFG.model_directory)
        self.model.load_state_dict(torch.load(filename))


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
