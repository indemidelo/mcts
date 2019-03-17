import numpy as np
from src.MCTS import SimulatedGame
from src.Player import Player
from src.NeuralNetwork import NeuralNetwork


class Training():
    def __init__(self, n_games, n_iter, n_moves, n_epochs, checkpoint=20):
        self.nn = NeuralNetwork()
        self.n_iter = n_iter
        self.n_moves = n_moves
        self.n_games = n_games
        self.n_epochs = n_epochs
        self.checkpoint = checkpoint
        self.p1, self.p2 = Player(1), Player(2)

    def play_and_train(self):
        for g in range(self.n_games):
            tau = 0.99 ** g
            simgame = SimulatedGame(
                self.p1, self.p2, self.n_iter,
                self.n_moves, tau=tau)
            training_raw_data = simgame.play_a_game()
            training_data = self.prepare_data(training_raw_data)
            self.nn.train(*training_data, self.n_epochs)
            if (g + 1) % self.checkpoint == 0:
                self.nn.save(g + 1)
    
    def test(self):
        from src.Board import Board
        from src.tfPlayer import tfPlayer
        from src.NNGame import NNRecordedGame
        b = Board()
        p1 = tfPlayer(1, b, self.nn.sess, self.nn.pred_policy, 
                      self.nn.inputs, training=False)
        p2 = tfPlayer(2, b, self.nn.sess, self.nn.pred_policy, 
                      self.nn.inputs, training=False)
        nn_g = NNRecordedGame(b, p1, p2, 1)
        nn_g.initialize()
        nn_g.play_a_game(print_board=True)

    def prepare_data(self, raw_data):
        input_data = np.array(raw_data['input']).reshape((-1, 6, 7, 3))
        output_data_pi = np.array(raw_data['pi'])
        output_data_z = np.array(raw_data['z']).reshape((-1, 1))
        return input_data, output_data_pi, output_data_z
