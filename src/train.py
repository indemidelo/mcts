import tensorflow as tf
import numpy as np
from src.MCTS import SimulatedGame
from src.Player import Player


class Training():
    def __init__(self, network, n_iter, n_moves, n_games, n_epochs, checkpoint=20):
        self.nn = network
        self.n_iter = n_iter
        self.n_moves = n_moves
        self.n_games = n_games
        self.n_epochs = n_epochs
        self.checkpoint = checkpoint
        self.p1, self.p2 = Player(1), Player(2)

    def play_and_train(self):
        for g in range(self.n_games):
            simgame = SimulatedGame(
                self.p1, self.p2, self.nn, self.n_iter, self.n_moves)
            simgame.initialize()
            training_raw_data = simgame.play_a_game()
            input_data, output_data = self.prepare_data(training_raw_data)
            self.nn.train(input_data, output_data, self.n_epochs)
            if g + 1 % self.checkpoint == 0:
                self.nn.save()

    def prepare_data(self, raw_data):
        input_data = np.asarray(raw_data['input'])
        output_data_pi = np.asarray(raw_data['output_pi'])
        output_data_z = np.asarray(raw_data['output_z'])
        return input_data, (output_data_pi, output_data_z)
