from config import CFG
import torch
import torch.optim as optim
from src.pytorch.network import AlphaGoNet, AlphaLoss


class NeuralNetwork(object):
    def __init__(self, game, model_name=None):
        self.game = game
        self.age = 0
        self.session_initialize()
        if model_name:
            self.load_model(model_name)

    def session_initialize(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = AlphaGoNet(self.game.input_shape()[0], self.game.policy_shape())
        # self.net.to(self.device)
        self.criterion = AlphaLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(), lr=CFG.learning_rate, momentum=CFG.momentum)

    def eval(self, state):
        board = state.board.board_as_tensor(state.player_color)
        board = to_tensor(board, self.device)
        p, v = self.net(board)
        p, v = to_array(p), to_array(v)
        return p[0], v[0]

    def train(self, input_data, output_data_pi, output_data_z):
        for epoch in range(CFG.epochs):
            self.optimizer.zero_grad()
            loss_policy_mean, loss_value_mean, loss_mean, j = .0, .0, .0, 0
            for i, pi, z in zip(input_data, output_data_pi, output_data_z):
                i = to_tensor(i, self.device)
                pi = to_tensor(pi, self.device)
                z = to_tensor(z, self.device)
                pred_pi, pred_z = self.net(i)
                loss, loss_value, loss_policy = self.criterion(z, pred_z, pi, pred_pi)
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


def to_tensor(array, device):
    tensor = torch.from_numpy(array)
    tensor = tensor.to(dtype=torch.float)
    # tensor = tensor.to(device, dtype=torch.float)
    return tensor


def to_array(tensor):
    array = tensor.cpu().detach().numpy()
    return array
